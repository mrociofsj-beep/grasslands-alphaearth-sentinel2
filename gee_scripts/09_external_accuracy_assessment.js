/*
External evaluation of a classified raster using reference points

This script:
1. Loads a previously generated classified raster
2. Loads an external evaluation point dataset
3. Standardizes class labels and maps them to class IDs
4. Extracts raster predictions at evaluation points
5. Computes confusion matrix-based accuracy metrics

Metrics reported:
- Overall Accuracy
- Kappa
- Producer's Accuracy
- User's Accuracy
- Macro-F1

Notes:
- The classified raster must have a band named 'class_id'
- The evaluation uses an external point dataset
*/

var YEAR = 2023;
var SETUP_NAME = 'AE__DEM_PHENO';
var SCALE = 10;

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var classifiedAsset = 'projects/your_project/assets/CLS_' + SETUP_NAME + '_' + YEAR + '_RF_final';
var evaluationPointsAsset = 'projects/your_project/assets/external_evaluation_points';
var evaluationClassProperty = 'class';

var ALLOWED_CLASS_IDS = [1, 2, 3, 4, 5, 6];
var classIds = ee.List(ALLOWED_CLASS_IDS);

var CLASS_MAP = ee.Dictionary({
  'rocky_grassland': 1,
  'grassland': 2,
  'cropland': 3,
  'water': 4,
  'urban': 5,
  'forest_plantation': 6
});

var CLASS_NAMES = ee.Dictionary({
  1: 'rocky_grassland',
  2: 'grassland',
  3: 'cropland',
  4: 'water',
  5: 'urban',
  6: 'forest_plantation'
});


// ---------------------------------------------------------------------
// 1. Load classified raster
// ---------------------------------------------------------------------
var classifiedRaster = ee.Image(classifiedAsset)
  .select(['class_id'])
  .rename('prediction')
  .toInt16()
  .clip(studyArea);


// ---------------------------------------------------------------------
// 2. Load and clean evaluation points
// ---------------------------------------------------------------------
var rawEvaluationPoints = ee.FeatureCollection(evaluationPointsAsset)
  .filterBounds(studyArea);


function normalizeClassLabel(classValue) {
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

    classLabel = normalizeClassLabel(classLabel);

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
  .filter(ee.Filter.inList('class_id', classIds));


// ---------------------------------------------------------------------
// 3. Sample raster predictions at evaluation points
// ---------------------------------------------------------------------
var sampledPoints = classifiedRaster.sampleRegions({
  collection: evaluationPoints,
  properties: ['class_id'],
  scale: SCALE,
  geometries: false
}).filter(ee.Filter.notNull(['prediction']));

var validSampledPoints = sampledPoints.filter(
  ee.Filter.inList('prediction', classIds)
);


// ---------------------------------------------------------------------
// 4. Confusion matrix and metrics
// ---------------------------------------------------------------------
var confusionMatrix = validSampledPoints.errorMatrix(
  'class_id',
  'prediction',
  classIds
);

var overallAccuracy = confusionMatrix.accuracy();
var kappa = confusionMatrix.kappa();


function flattenAccuracyArray(arrayLike) {
  return ee.List(ee.Array(arrayLike).toList()).flatten();
}

var producerAccuracy = flattenAccuracyArray(confusionMatrix.producersAccuracy());
var userAccuracy = flattenAccuracyArray(confusionMatrix.consumersAccuracy());

var f1PerClass = ee.List.sequence(0, classIds.length().subtract(1)).map(function(i) {
  i = ee.Number(i);

  var recall = ee.Number(producerAccuracy.get(i));
  var precision = ee.Number(userAccuracy.get(i));
  var denominator = precision.add(recall);

  return ee.Number(
    ee.Algorithms.If(
      denominator.neq(0),
      precision.multiply(recall).multiply(2).divide(denominator),
      0
    )
  );
});

var macroF1 = ee.Number(f1PerClass.reduce(ee.Reducer.mean()));


// ---------------------------------------------------------------------
// 5. Build readable per-class summary
// ---------------------------------------------------------------------
var perClassTable = classIds.map(function(classId) {
  classId = ee.Number(classId);
  var index = classIds.indexOf(classId);

  return ee.Feature(null, {
    class_id: classId,
    class_name: CLASS_NAMES.get(classId),
    producer_accuracy: producerAccuracy.get(index),
    user_accuracy: userAccuracy.get(index),
    f1_score: f1PerClass.get(index)
  });
});


// ---------------------------------------------------------------------
// 6. Print results
// ---------------------------------------------------------------------
print('Confusion matrix:', confusionMatrix);
print('Overall Accuracy:', overallAccuracy);
print('Kappa:', kappa);
print('Macro-F1:', macroF1);
print('Per-class metrics:', ee.FeatureCollection(perClassTable));