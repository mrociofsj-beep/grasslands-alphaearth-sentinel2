/*
Generate the final land-cover classification map using the selected
Random Forest configuration and the full pool of valid training points.

This script:
1. Loads the predictor layers for a selected setup
2. Loads the full pool of valid training points
3. Optionally applies simple class balancing
4. Trains the final Random Forest classifier
5. Classifies the predictor stack
6. Exports the classified map
7. Optionally exports class probabilities

Notes:
- This script generates the final classified raster
- Training uses the full valid point pool for the selected year
- Optional balancing is applied only if explicitly enabled
*/

var YEAR = 2023;
var SCALE = 10;
var NODATA = -9999;
var RANDOM_SEED = 2025;
var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var studyAreaMask = ee.Image.constant(1)
  .clip(studyArea)
  .reproject({crs: REF_CRS, scale: SCALE});


// ---------------------------------------------------------------------
// 1. Random Forest parameters
// ---------------------------------------------------------------------
// These values should match the optimal hyperparameters obtained
// for the selected predictor setup from the hyperparameter search step.
var RF_PARAMETERS = {
  numberOfTrees: 500,
  bagFraction: 0.60,
  minLeafPopulation: 1
};

// Number of variables randomly selected at each split (mtry).
// This value may vary depending on the selected predictor setup.
var MTRY_FIXED = 9;


// ---------------------------------------------------------------------
// 2. Export settings
// ---------------------------------------------------------------------
var EXPORT_TO_ASSET = true;
var EXPORT_TO_DRIVE = true;
var EXPORT_PROBABILITIES = false;

var ASSET_BASE = 'projects/your_project/assets';
var DRIVE_FOLDER_CLASSIFIED = 'classified_maps';


// ---------------------------------------------------------------------
// 3. Optional class balancing
// ---------------------------------------------------------------------
var USE_BALANCED_TRAINING = false;
var MAX_TRAIN_POINTS_PER_CLASS = 4000;


// ---------------------------------------------------------------------
// 4. Class definitions
// ---------------------------------------------------------------------
var ALLOWED_CLASS_IDS = [1, 2, 3, 4, 5, 6];
var CLASS_IDS = ee.List(ALLOWED_CLASS_IDS);

var CLASS_NAMES = [
  'rocky_grassland',
  'grassland',
  'cropland',
  'water',
  'urban',
  'forest_plantation'
];


// ---------------------------------------------------------------------
// 5. Predictor assets
// ---------------------------------------------------------------------
var s2Assets = [
  'projects/your_project/assets/S2_2023-01-01_to_2023-04-01',
  'projects/your_project/assets/S2_2023-04-01_to_2023-07-01',
  'projects/your_project/assets/S2_2023-07-01_to_2023-10-01',
  'projects/your_project/assets/S2_2023-10-01_to_2023-12-31'
];

var demAssets = [
  'projects/your_project/assets/DEM_stack_10m'
];

var phenoAssets = [
  'projects/your_project/assets/PHENO_2023-01-01_to_2023-04-01',
  'projects/your_project/assets/PHENO_2023-04-01_to_2023-07-01',
  'projects/your_project/assets/PHENO_2023-07-01_to_2023-10-01',
  'projects/your_project/assets/PHENO_2023-10-01_to_2023-12-31'
];

var aeAssets = [
  'projects/your_project/assets/AE_2023'
];


// ---------------------------------------------------------------------
// 6. Selected predictor setup
// ---------------------------------------------------------------------
var SETUP = {
  name: 'AE_PHENO',
  useS2: false,
  useDEM: false,
  usePHENO: true,
  useAE: true
};


// ---------------------------------------------------------------------
// 7. Stack utilities
// ---------------------------------------------------------------------
function loadAndPrepareImage(assetId) {
  return ee.Image(assetId)
    .unmask(NODATA)
    .updateMask(studyAreaMask);
}


function prefixBandNames(image, prefix) {
  image = ee.Image(image);
  var bandNames = image.bandNames();

  return image.rename(
    bandNames.map(function(bandName) {
      return ee.String(prefix).cat(ee.String(bandName));
    })
  );
}


function stackFromAssets(assetList, groupPrefix) {
  var imageList = [];

  for (var i = 0; i < assetList.length; i++) {
    var image = loadAndPrepareImage(assetList[i]);

    var prefix;
    if (groupPrefix === 'DEM' || groupPrefix === 'AE') {
      prefix = groupPrefix + '__';
    } else {
      prefix = groupPrefix + '_Q' + ('0' + (i + 1)).slice(-2) + '__';
    }

    imageList.push(prefixBandNames(image, prefix));
  }

  var stacked = ee.Image(imageList[0]);
  for (var j = 1; j < imageList.length; j++) {
    stacked = stacked.addBands(imageList[j], null, true);
  }

  return stacked;
}


var s2Stack = SETUP.useS2 ? stackFromAssets(s2Assets, 'S2') : null;
var demStack = SETUP.useDEM ? stackFromAssets(demAssets, 'DEM') : null;
var phenoStack = SETUP.usePHENO ? stackFromAssets(phenoAssets, 'PH') : null;
var aeStack = SETUP.useAE ? stackFromAssets(aeAssets, 'AE') : null;


function buildPredictorStack(setup) {
  var image = ee.Image(1).select([]);

  if (setup.useS2) {
    image = image.addBands(s2Stack, null, true);
  }
  if (setup.useAE) {
    image = image.addBands(aeStack, null, true);
  }
  if (setup.useDEM) {
    image = image.addBands(demStack, null, true);
  }
  if (setup.usePHENO) {
    image = image.addBands(phenoStack, null, true);
  }

  return image.updateMask(studyAreaMask).toFloat();
}


var predictorStack = buildPredictorStack(SETUP);
var stackBands = predictorStack.bandNames();


// ---------------------------------------------------------------------
// 8. Training data
// ---------------------------------------------------------------------
var trainingAssetId = ASSET_BASE + '/TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK';
var allTrainingPoints = ee.FeatureCollection(trainingAssetId);

var firstFeature = allTrainingPoints.first();
var allProperties = ee.List(firstFeature.propertyNames());

var candidateBands = allProperties.removeAll([
  'system:index',
  '.geo',
  'class_id',
  'rand'
]);

var bandList = candidateBands.filter(
  ee.Filter.inList('item', stackBands)
);

var trainingRaw = allTrainingPoints
  .filter(ee.Filter.inList('class_id', ALLOWED_CLASS_IDS))
  .select(bandList.cat(['class_id']));


function balanceByCap(featureCollection, maxPerClass) {
  featureCollection = ee.FeatureCollection(featureCollection);

  var classIds = ee.List(featureCollection.aggregate_array('class_id').distinct());

  var perClassCollections = classIds.map(function(classId) {
    classId = ee.Number(classId);

    return featureCollection
      .filter(ee.Filter.eq('class_id', classId))
      .randomColumn('rand_bal', RANDOM_SEED)
      .sort('rand_bal')
      .limit(maxPerClass);
  });

  return ee.FeatureCollection(perClassCollections).flatten();
}


var trainingSet = ee.FeatureCollection(
  ee.Algorithms.If(
    USE_BALANCED_TRAINING,
    balanceByCap(trainingRaw, MAX_TRAIN_POINTS_PER_CLASS),
    trainingRaw
  )
);


// ---------------------------------------------------------------------
// 9. Train final Random Forest model
// ---------------------------------------------------------------------
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: RF_PARAMETERS.numberOfTrees,
  variablesPerSplit: MTRY_FIXED,
  minLeafPopulation: RF_PARAMETERS.minLeafPopulation,
  bagFraction: RF_PARAMETERS.bagFraction,
  seed: RANDOM_SEED
}).train({
  features: trainingSet,
  classProperty: 'class_id',
  inputProperties: bandList
});


// ---------------------------------------------------------------------
// 10. Classify raster
// ---------------------------------------------------------------------
var classifiedMap = predictorStack
  .select(bandList)
  .classify(classifier)
  .rename('class_id')
  .int16()
  .set({
    class_values: [1, 2, 3, 4, 5, 6],
    class_names: CLASS_NAMES
  });

var outputName = 'CLS_' + SETUP.name + '_' + YEAR + '_RF_final';


// ---------------------------------------------------------------------
// 11. Export classified map
// ---------------------------------------------------------------------
if (EXPORT_TO_ASSET) {
  Export.image.toAsset({
    image: classifiedMap,
    description: outputName,
    assetId: ASSET_BASE + '/' + outputName,
    region: studyArea,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });
}

if (EXPORT_TO_DRIVE) {
  Export.image.toDrive({
    image: classifiedMap.updateMask(studyAreaMask),
    description: outputName + '_DRIVE',
    folder: DRIVE_FOLDER_CLASSIFIED,
    fileNamePrefix: outputName,
    region: studyArea,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });
}


// ---------------------------------------------------------------------
// 12. Optional probability export
// ---------------------------------------------------------------------
if (EXPORT_PROBABILITIES) {
  var probabilityImage = predictorStack
    .select(bandList)
    .classify(classifier, 'probability');

  var probabilityBandNames = CLASS_IDS.map(function(classId) {
    classId = ee.Number(classId);
    return ee.String('probability_').cat(classId.format());
  });

  Export.image.toAsset({
    image: probabilityImage.select(probabilityBandNames),
    description: outputName + '_PROB',
    assetId: ASSET_BASE + '/' + outputName + '_PROB',
    region: studyArea,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });
}


print('Final classification export prepared. Run the tasks from the Tasks tab.');