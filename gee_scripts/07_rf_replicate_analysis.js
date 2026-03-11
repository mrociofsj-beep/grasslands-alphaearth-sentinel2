/*
Random Forest replicate analysis with fixed external evaluation

This script:
1. Repeats Random Forest model fitting for a selected predictor setup
2. Generates a different stratified training subset in each replicate
3. Keeps the external evaluation dataset fixed across all replicates
4. Computes performance metrics for each replicate
5. Exports replicate-level results as a CSV table

Metrics exported:
- setup
- replicate
- seed
- OA
- Kappa
- MacroF1
- PA_*
- UA_*
- nEvalUsed
- nTrainUsed
- nTrain_1 ... nTrain_6

Notes:
- No classification maps are exported
- Training subsets vary across replicates
- External evaluation remains fixed
*/

var YEAR = 2023;
var SCALE = 10;
var REF_CRS = 'EPSG:32721';
var NODATA = -9999;

var BASE_SEED = 2025;
var N_REPLICATES = 100;

// Random Forest hyperparameters.
// These values should correspond to the optimal configuration
// obtained from the hyperparameter search for the selected predictor setup.
// Users should update these values according to the results of
// the hyperparameter optimisation step.
var RF_PARAMETERS = {
  numberOfTrees: 500,
  bagFraction: 0.60,
  minLeafPopulation: 1
};

// Number of variables randomly selected at each split (mtry).
// This value may vary depending on the number of predictor variables
// used in the selected setup.
var MTRY_FIXED = 9;

var TRAIN_FRACTION = 0.70;
var MIN_TRAIN_PER_CLASS = 150;
var MAX_TRAIN_PER_CLASS = 4000;

var DRIVE_FOLDER = 'rf_replicate_metrics';


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
// 2. Class definitions
// ---------------------------------------------------------------------
var CLASS_MAP = ee.Dictionary({
  'rocky_grassland': 1,
  'grassland': 2,
  'cropland': 3,
  'water': 4,
  'urban': 5,
  'forest_plantation': 6
});

var ALLOWED_CLASS_IDS = [1, 2, 3, 4, 5, 6];
var CLASS_IDS = ee.List(ALLOWED_CLASS_IDS);


// ---------------------------------------------------------------------
// 3. Predictor assets
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
// 4. Selected predictor setup
// ---------------------------------------------------------------------
var SETUP = {
  name: 'AE_DEM_PHENO',
  useS2: false,
  useDEM: true,
  usePHENO: true,
  useAE: true
};


// ---------------------------------------------------------------------
// 5. Stack utilities
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


var s2Stack = stackFromAssets(s2Assets, 'S2');
var demStack = stackFromAssets(demAssets, 'DEM');
var phenoStack = stackFromAssets(phenoAssets, 'PH');
var aeStack = stackFromAssets(aeAssets, 'AE');


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


// ---------------------------------------------------------------------
// 6. Base training dataset
// ---------------------------------------------------------------------
var assetBase = 'projects/your_project/assets';
var trainingAssetId = assetBase + '/TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK';

var allTrainingPoints = ee.FeatureCollection(trainingAssetId);

var trainingPool = allTrainingPoints
  .filter(ee.Filter.inList('class_id', ALLOWED_CLASS_IDS));


// ---------------------------------------------------------------------
// 7. Build replicate-specific training subset
// ---------------------------------------------------------------------
function buildStratifiedTrainingSubset(featureCollection, trainFraction, maxPerClass, minPerClass, seed) {
  featureCollection = ee.FeatureCollection(featureCollection);
  trainFraction = ee.Number(trainFraction);
  maxPerClass = ee.Number(maxPerClass);
  minPerClass = ee.Number(minPerClass);

  var perClassCollections = CLASS_IDS.map(function(classId) {
    classId = ee.Number(classId);

    var classSubset = featureCollection.filter(ee.Filter.eq('class_id', classId));

    classSubset = classSubset.randomColumn('rand_split', seed.add(classId));
    var trainingSubset = classSubset.filter(ee.Filter.lt('rand_split', trainFraction));

    trainingSubset = ee.FeatureCollection(
      ee.Algorithms.If(
        trainingSubset.size().gte(minPerClass),
        trainingSubset,
        classSubset
      )
    );

    trainingSubset = trainingSubset
      .randomColumn('rand_cap', seed.add(classId).add(999))
      .sort('rand_cap')
      .limit(maxPerClass);

    return trainingSubset;
  });

  return ee.FeatureCollection(perClassCollections).flatten();
}


// ---------------------------------------------------------------------
// 8. Fixed external evaluation dataset
// ---------------------------------------------------------------------
var evaluationClassProperty = 'class';
var evaluationPointsAsset = 'projects/your_project/assets/external_evaluation_points';

var rawEvaluationPoints = ee.FeatureCollection(evaluationPointsAsset)
  .filterBounds(studyArea);


function normalizeEvaluationClassLabel(classValue) {
  classValue = ee.String(classValue).trim().toLowerCase();

  classValue = classValue
    .replace('á', 'a')
    .replace('é', 'e')
    .replace('í', 'i')
    .replace('ó', 'o')
    .replace('ú', 'u');

  classValue = ee.String(
    ee.Algorithms.If(
      classValue.equals('rocky grassland'),
      'rocky_grassland',
      classValue
    )
  );

  classValue = ee.String(
    ee.Algorithms.If(
      classValue.equals('forestall'),
      'forest_plantation',
      classValue
    )
  );

  return classValue;
}


var evaluationPoints = rawEvaluationPoints
  .map(function(feature) {
    var hasClass = feature.propertyNames().contains(evaluationClassProperty);

    var classLabel = ee.String(
      ee.Algorithms.If(hasClass, feature.get(evaluationClassProperty), '')
    );

    classLabel = normalizeEvaluationClassLabel(classLabel);

    var classId = ee.Number(
      ee.Algorithms.If(
        CLASS_MAP.contains(classLabel),
        CLASS_MAP.get(classLabel),
        -99
      )
    );

    return ee.Feature(feature.geometry(), {
      class_id: classId
    });
  })
  .filter(ee.Filter.neq('class_id', -99))
  .filter(ee.Filter.inList('class_id', CLASS_IDS));


// ---------------------------------------------------------------------
// 9. Metric helpers
// ---------------------------------------------------------------------
function flatten1D(arrayLike) {
  var array = ee.Array(arrayLike);
  var list2d = ee.List(array.toList());
  var first = ee.List(list2d.get(0));

  var isColumnVector = ee.Algorithms.IsEqual(first.length(), 1);

  return ee.List(
    ee.Algorithms.If(
      isColumnVector,
      list2d.map(function(x) {
        return ee.Number(ee.List(x).get(0));
      }),
      first
    )
  );
}


function safeF1(precision, recall) {
  precision = ee.Number(precision);
  recall = ee.Number(recall);

  var denominator = precision.add(recall);

  return ee.Number(
    ee.Algorithms.If(
      denominator.neq(0),
      precision.multiply(recall).multiply(2).divide(denominator),
      0
    )
  );
}


// ---------------------------------------------------------------------
// 10. Run one replicate
// ---------------------------------------------------------------------
function runOneReplicate(setupName, predictorStack, replicateId) {
  replicateId = ee.Number(replicateId);
  var replicateSeed = ee.Number(BASE_SEED).add(replicateId);

  var stackBands = predictorStack.bandNames();

  var firstFeature = trainingPool.first();
  var allProperties = ee.List(firstFeature.propertyNames());

  var candidateBands = allProperties.removeAll([
    'system:index',
    '.geo',
    'class_id',
    'rand',
    'rand_bal',
    'rand_split',
    'rand_cap'
  ]);

  var bandList = candidateBands.filter(ee.Filter.inList('item', stackBands));

  var trainingTable = trainingPool.select(bandList.cat(['class_id']));

  var replicateTrainingSet = buildStratifiedTrainingSubset(
    trainingTable,
    TRAIN_FRACTION,
    MAX_TRAIN_PER_CLASS,
    MIN_TRAIN_PER_CLASS,
    replicateSeed
  );

  var trainingCounts = CLASS_IDS.map(function(classId) {
    classId = ee.Number(classId);
    return replicateTrainingSet.filter(ee.Filter.eq('class_id', classId)).size();
  });

  var nTrainUsed = replicateTrainingSet.size();

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: RF_PARAMETERS.numberOfTrees,
    variablesPerSplit: MTRY_FIXED,
    minLeafPopulation: RF_PARAMETERS.minLeafPopulation,
    bagFraction: RF_PARAMETERS.bagFraction,
    seed: replicateSeed
  }).train({
    features: replicateTrainingSet,
    classProperty: 'class_id',
    inputProperties: bandList
  });

  var evaluationTable = predictorStack.select(bandList).sampleRegions({
    collection: evaluationPoints,
    properties: ['class_id'],
    scale: SCALE,
    geometries: false
  }).filter(ee.Filter.notNull(bandList));

  var predictedTable = evaluationTable.classify(classifier).map(function(feature) {
    return feature.set('prediction', feature.get('classification'));
  });

  var validPredictions = predictedTable.filter(ee.Filter.inList('prediction', CLASS_IDS));
  var confusionMatrix = validPredictions.errorMatrix('class_id', 'prediction', CLASS_IDS);

  var overallAccuracy = confusionMatrix.accuracy();
  var kappa = confusionMatrix.kappa();

  var producerAccuracy = flatten1D(ee.Array(confusionMatrix.producersAccuracy()));
  var userAccuracy = flatten1D(ee.Array(confusionMatrix.consumersAccuracy()));

  var f1PerClass = CLASS_IDS.map(function(classId) {
    classId = ee.Number(classId);
    var index = CLASS_IDS.indexOf(classId);
    return safeF1(
      ee.Number(userAccuracy.get(index)),
      ee.Number(producerAccuracy.get(index))
    );
  });

  var macroF1 = ee.Number(ee.List(f1PerClass).reduce(ee.Reducer.mean()));

  var paUaProperties = ee.Dictionary(
    CLASS_IDS.iterate(function(classId, acc) {
      classId = ee.Number(classId);
      acc = ee.Dictionary(acc);
      var index = CLASS_IDS.indexOf(classId);

      return acc
        .set(ee.String('PA_').cat(classId.format()), ee.Number(producerAccuracy.get(index)))
        .set(ee.String('UA_').cat(classId.format()), ee.Number(userAccuracy.get(index)));
    }, ee.Dictionary({}))
  );

  var trainCountProperties = ee.Dictionary(
    CLASS_IDS.iterate(function(classId, acc) {
      classId = ee.Number(classId);
      acc = ee.Dictionary(acc);
      var index = CLASS_IDS.indexOf(classId);

      return acc.set(
        ee.String('nTrain_').cat(classId.format()),
        ee.Number(trainingCounts.get(index))
      );
    }, ee.Dictionary({}))
  );

  var baseProperties = ee.Dictionary({
    setup: setupName,
    replicate: replicateId,
    seed: replicateSeed,
    nEvalUsed: validPredictions.size(),
    nTrainUsed: nTrainUsed,
    OA: overallAccuracy,
    Kappa: kappa,
    MacroF1: macroF1
  });

  return ee.Feature(null, baseProperties)
    .setMulti(paUaProperties)
    .setMulti(trainCountProperties);
}


// ---------------------------------------------------------------------
// 11. Run all replicates
// ---------------------------------------------------------------------
var predictorStack = buildPredictorStack(SETUP);

var replicateIds = ee.List.sequence(1, N_REPLICATES);

var replicateResults = ee.FeatureCollection(
  replicateIds.map(function(replicateId) {
    return runOneReplicate(SETUP.name, predictorStack, replicateId);
  })
);


// ---------------------------------------------------------------------
// 12. Export replicate metrics
// ---------------------------------------------------------------------
Export.table.toDrive({
  collection: replicateResults,
  description: 'RF_replicates_' + SETUP.name + '_' + YEAR,
  folder: DRIVE_FOLDER,
  fileNamePrefix: 'RF_replicates_' + SETUP.name + '_' + YEAR,
  fileFormat: 'CSV'
});


print('Random Forest replicate analysis prepared. Run the export task from the Tasks tab.');