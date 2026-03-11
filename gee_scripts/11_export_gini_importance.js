/*
Export impurity-based Random Forest variable importance (Gini importance)

This script:
1. Loads the predictor layers for a selected setup
2. Loads the full pool of valid reference points
3. Trains one Random Forest model per seed
4. Extracts impurity-based variable importance from each model
5. Exports one CSV table containing feature importance values for all seeds

Exports:
- FI_Gini_<SETUP>_<YEAR>_seeds_<START>_<END>.csv

Notes:
- The script uses the full valid training pool for the selected setup
- One model is trained per seed
- The exported table is intended for downstream summary analyses in Python
*/

var YEAR = 2023;
var SCALE = 10;
var NODATA = -9999;
var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var SEED_START = 0;
var SEED_END = 99;

var DRIVE_FOLDER_GINI = 'gini_importance_exports';


// ---------------------------------------------------------------------
// 1. Study area
// ---------------------------------------------------------------------
var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var studyAreaMask = ee.Image.constant(1)
  .clip(studyArea)
  .reproject({crs: REF_CRS, scale: SCALE});


// ---------------------------------------------------------------------
// 2. Selected predictor setup
// ---------------------------------------------------------------------
// Available setup names:
// S2, S2_DEM, S2_PHENO, S2_DEM_PHENO,
// AE, AE_DEM, AE_PHENO, AE_DEM_PHENO
var SETUP = {
  name: 'S2_DEM_PHENO',
  useS2: true,
  useDEM: true,
  usePHENO: true,
  useAE: false
};


// ---------------------------------------------------------------------
// 3. Random Forest hyperparameters by setup
// ---------------------------------------------------------------------
// Keys must match SETUP.name exactly.
var RF_PARAMS_BY_SETUP = {
  'S2':           {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 8},
  'S2_DEM':       {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 7},
  'S2_PHENO':     {numberOfTrees: 300, bagFraction: 0.60, minLeafPopulation: 1, mtry: 7},
  'S2_DEM_PHENO': {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 8},
  'AE':           {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 5},
  'AE_DEM':       {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 5},
  'AE_PHENO':     {numberOfTrees: 300, bagFraction: 0.60, minLeafPopulation: 1, mtry: 5},
  'AE_DEM_PHENO': {numberOfTrees: 500, bagFraction: 0.60, minLeafPopulation: 1, mtry: 9}
};

function getRFParamsForSetup(setupName) {
  var params = RF_PARAMS_BY_SETUP[setupName];

  if (!params) {
    throw new Error(
      'No Random Forest hyperparameters were found for setup: ' + setupName
    );
  }

  return params;
}

var RF_PARAMS = getRFParamsForSetup(SETUP.name);


// ---------------------------------------------------------------------
// 4. Class definitions
// ---------------------------------------------------------------------
var ALLOWED_CLASS_IDS = [1, 2, 3, 4, 5, 6];


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
// 6. Optional class balancing
// ---------------------------------------------------------------------
var USE_BALANCED_TRAINING = false;
var MAX_TRAIN_POINTS_PER_CLASS = 4000;


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
// 8. Training table
// ---------------------------------------------------------------------
var ASSET_BASE = 'projects/your_project/assets';
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

var bandList = stackBands.filter(
  ee.Filter.inList('item', candidateBands)
);

var trainingRaw = allTrainingPoints
  .filter(ee.Filter.inList('class_id', ALLOWED_CLASS_IDS))
  .select(bandList.cat(['class_id']));

function balanceByCap(featureCollection, maxPerClass, seed) {
  featureCollection = ee.FeatureCollection(featureCollection);

  var classIds = ee.List(featureCollection.aggregate_array('class_id').distinct());

  var perClassCollections = classIds.map(function(classId) {
    classId = ee.Number(classId);

    return featureCollection
      .filter(ee.Filter.eq('class_id', classId))
      .randomColumn('rand_bal', seed)
      .sort('rand_bal')
      .limit(maxPerClass);
  });

  return ee.FeatureCollection(perClassCollections).flatten();
}


// ---------------------------------------------------------------------
// 9. Train one model and extract Gini importance
// ---------------------------------------------------------------------
function trainAndExtractGini(seed) {
  seed = ee.Number(seed);

  var trainingSet = ee.FeatureCollection(
    ee.Algorithms.If(
      USE_BALANCED_TRAINING,
      balanceByCap(trainingRaw, MAX_TRAIN_POINTS_PER_CLASS, seed),
      trainingRaw
    )
  );

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: RF_PARAMS.numberOfTrees,
    variablesPerSplit: RF_PARAMS.mtry,
    minLeafPopulation: RF_PARAMS.minLeafPopulation,
    bagFraction: RF_PARAMS.bagFraction,
    seed: seed
  }).train({
    features: trainingSet,
    classProperty: 'class_id',
    inputProperties: bandList
  });

  var explanation = ee.Dictionary(classifier.explain());
  var importance = ee.Dictionary(explanation.get('importance'));
  var outOfBagError = ee.Number(explanation.get('outOfBagErrorEstimate'));

  var features = importance.keys();

  return ee.FeatureCollection(
    features.map(function(featureName) {
      featureName = ee.String(featureName);

      return ee.Feature(null, {
        setup: SETUP.name,
        year: YEAR,
        seed: seed,
        oob_error: outOfBagError,
        feature: featureName,
        gini_importance: ee.Number(importance.get(featureName))
      });
    })
  );
}


// ---------------------------------------------------------------------
// 10. Run all seeds
// ---------------------------------------------------------------------
var seeds = ee.List.sequence(SEED_START, SEED_END);

var allGiniResults = ee.FeatureCollection(
  seeds.map(function(seed) {
    return trainAndExtractGini(seed);
  })
).flatten();


// ---------------------------------------------------------------------
// 11. Export results
// ---------------------------------------------------------------------
Export.table.toDrive({
  collection: allGiniResults,
  description: 'FI_Gini_' + SETUP.name + '_' + YEAR + '_seeds_' + SEED_START + '_' + SEED_END,
  folder: DRIVE_FOLDER_GINI,
  fileNamePrefix: 'FI_Gini_' + SETUP.name + '_' + YEAR + '_seeds_' + SEED_START + '_' + SEED_END,
  fileFormat: 'CSV'
});


print('Gini importance export prepared. Run the export task from the Tasks tab.');