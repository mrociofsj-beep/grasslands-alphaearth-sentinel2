/*
Random Forest hyperparameter search for a selected predictor setup

This script:
1. Loads precomputed training and test tables for a selected predictor setup
2. Defines a grid of Random Forest hyperparameters
3. Trains and evaluates one model per parameter combination
4. Computes Overall Accuracy and Macro-F1 on the test set
5. Stores the results as a FeatureCollection
6. Optionally exports the results as a CSV table

Expected input assets:
- TRAIN_BALANCED_<YEAR>_<SETUP_NAME>
- TEST_RAW_<YEAR>_<SETUP_NAME>

Available setup names:
- S2
- S2_DEM
- S2_PHENO
- S2_DEM_PHENO
- AE
- AE_DEM
- AE_PHENO
- AE_DEM_PHENO
*/

var YEAR = 2023;
var SETUP_NAME = 'S2_PHENO';
var RANDOM_SEED = 2025;
var MAX_TEST_SAMPLES = 6000;


// ---------------------------------------------------------------------
// 1. Hyperparameter grid
// ---------------------------------------------------------------------
var N_TREES_LIST = ee.List([300, 400, 500]);
var BAG_FRACTION_LIST = ee.List([0.5, 0.6]);
var MIN_LEAF_LIST = ee.List([1, 3]);
var MTRY_LIST = ee.List([5, 7, 9]);


// ---------------------------------------------------------------------
// 2. Load prepared datasets
// ---------------------------------------------------------------------
var ASSET_BASE = 'projects/your_project/assets/';

var trainAssetId = ASSET_BASE + 'TRAIN_BALANCED_' + YEAR + '_' + SETUP_NAME;
var testAssetId = ASSET_BASE + 'TEST_RAW_' + YEAR + '_' + SETUP_NAME;

var trainingSet = ee.FeatureCollection(trainAssetId);
var testSetRaw = ee.FeatureCollection(testAssetId);


// ---------------------------------------------------------------------
// 3. Class order and input predictors
// ---------------------------------------------------------------------
var classOrder = ee.List(
  trainingSet.aggregate_array('class_id')
    .distinct()
    .sort()
);

var firstFeature = trainingSet.first();
var allProperties = ee.List(firstFeature.propertyNames());

var inputBands = allProperties.removeAll([
  'system:index',
  '.geo',
  'class_id',
  'rand'
]);


// ---------------------------------------------------------------------
// 4. Optionally reduce test set size
// ---------------------------------------------------------------------
var hasRandomColumn = testSetRaw.first().propertyNames().contains('rand');

var testSet = ee.FeatureCollection(
  ee.Algorithms.If(
    hasRandomColumn,
    testSetRaw.sort('rand').limit(MAX_TEST_SAMPLES),
    testSetRaw.limit(MAX_TEST_SAMPLES)
  )
);


// ---------------------------------------------------------------------
// 5. Helper functions
// ---------------------------------------------------------------------
function to1DList(arrayLike) {
  return ee.List(ee.Array(arrayLike).toList()).flatten();
}


function evaluateRandomForest(params) {
  params = ee.Dictionary(params);

  var nTrees = ee.Number(params.get('n_trees'));
  var bagFraction = ee.Number(params.get('bag_fraction'));
  var minLeaf = ee.Number(params.get('min_leaf'));
  var mtry = ee.Number(params.get('mtry'));

  var nBands = ee.Number(inputBands.size());
  var mtryClamped = mtry.max(1).min(nBands).int();

  var classifier = ee.Classifier.smileRandomForest({
    numberOfTrees: nTrees,
    variablesPerSplit: mtryClamped,
    minLeafPopulation: minLeaf,
    bagFraction: bagFraction,
    seed: RANDOM_SEED
  }).train({
    features: trainingSet,
    classProperty: 'class_id',
    inputProperties: inputBands
  });

  var classifiedTest = testSet.classify(classifier);
  var confusionMatrix = classifiedTest.errorMatrix('class_id', 'classification', classOrder);

  var overallAccuracy = confusionMatrix.accuracy();

  var precisionList = to1DList(confusionMatrix.consumersAccuracy());
  var recallList = to1DList(confusionMatrix.producersAccuracy());

  var nClasses = ee.Number(
    ee.List([precisionList.size(), recallList.size()])
      .reduce(ee.Reducer.min())
  );

  var f1List = ee.List.sequence(0, nClasses.subtract(1)).map(function(i) {
    i = ee.Number(i);

    var precision = ee.Number(precisionList.get(i));
    var recall = ee.Number(recallList.get(i));
    var denominator = precision.add(recall);

    return ee.Number(
      ee.Algorithms.If(
        denominator.gt(0),
        precision.multiply(recall).multiply(2).divide(denominator),
        0
      )
    );
  });

  var macroF1 = ee.Number(
    ee.Array(f1List).reduce('mean', [0]).get([0])
  );

  return ee.Dictionary({
    n_trees: nTrees,
    bag_fraction: bagFraction,
    min_leaf: minLeaf,
    mtry: mtryClamped,
    OA: overallAccuracy,
    MacroF1: macroF1
  });
}


// ---------------------------------------------------------------------
// 6. Build parameter grid
// ---------------------------------------------------------------------
var parameterGrid = N_TREES_LIST.map(function(nTrees) {
  return BAG_FRACTION_LIST.map(function(bagFraction) {
    return MIN_LEAF_LIST.map(function(minLeaf) {
      return MTRY_LIST.map(function(mtry) {
        return ee.Dictionary({
          n_trees: nTrees,
          bag_fraction: bagFraction,
          min_leaf: minLeaf,
          mtry: mtry
        });
      });
    }).flatten();
  }).flatten();
}).flatten();


// ---------------------------------------------------------------------
// 7. Evaluate all combinations
// ---------------------------------------------------------------------
var results = ee.FeatureCollection(
  parameterGrid.map(function(params) {
    return ee.Feature(null, evaluateRandomForest(params));
  })
);


// ---------------------------------------------------------------------
// 8. Rank results
// ---------------------------------------------------------------------
var resultsByMacroF1 = results.sort('MacroF1', false);
var resultsByOA = results.sort('OA', false);


// ---------------------------------------------------------------------
// 9. Optional export
// ---------------------------------------------------------------------
/*
Export.table.toDrive({
  collection: results,
  description: 'RF_hyperparameter_search_' + YEAR + '_' + SETUP_NAME,
  folder: 'rf_hyperparameter_search',
  fileNamePrefix: 'RF_hyperparameter_search_' + YEAR + '_' + SETUP_NAME,
  fileFormat: 'CSV'
});
*/


print('Random Forest hyperparameter search completed.');
print('Results:', results);
print('Top models by MacroF1:', resultsByMacroF1.limit(10));
print('Top models by OA:', resultsByOA.limit(10));