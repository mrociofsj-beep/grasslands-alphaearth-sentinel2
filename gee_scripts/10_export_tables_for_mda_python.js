/*
Export training-pool and external evaluation tables for MDA analysis in Python

This script:
1. Loads the predictor layers for a selected setup
2. Loads the full pool of valid reference points
3. Selects the predictor bands corresponding to the chosen setup
4. Exports a training pool table for replicate resampling in Python
5. Exports a fixed external evaluation table for permutation-based MDA analysis

Exports:
- TRAINPOOL_<SETUP>_<YEAR>.csv
- EVAL_<SETUP>_<YEAR>.csv

Notes:
- The training pool contains all valid reference points for the selected setup
- The evaluation table is built from the fixed external evaluation dataset
- These exports are intended for downstream MDA analyses in Python
*/

var YEAR = 2023;
var SCALE = 10;
var NODATA = -9999;
var REF_CRS = 'EPSG:32721';

var DRIVE_FOLDER_MDA = 'mda_python_tables';


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
// Available setup names:
// S2, S2_DEM, S2_PHENO, S2_DEM_PHENO,
// AE, AE_DEM, AE_PHENO, AE_DEM_PHENO
var SETUP = {
  name: 'AE_DEM_PHENO',
  useS2: false,
  useDEM: true,
  usePHENO: true,
  useAE: true
};


// ---------------------------------------------------------------------
// 5. Input tables
// ---------------------------------------------------------------------
var ASSET_BASE = 'projects/your_project/assets';

var trainingPoolAsset = ASSET_BASE + '/TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK';
var evaluationPointsAsset = ASSET_BASE + '/external_evaluation_points';
var evaluationClassProperty = 'class';


// ---------------------------------------------------------------------
// 6. Stack utilities
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

var predictorStack = buildPredictorStack(SETUP);
var stackBands = predictorStack.bandNames();


// ---------------------------------------------------------------------
// 7. Training pool table
// ---------------------------------------------------------------------
var allTrainingPoints = ee.FeatureCollection(trainingPoolAsset)
  .filter(ee.Filter.inList('class_id', ALLOWED_CLASS_IDS));

var firstFeature = allTrainingPoints.first();
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

var bandList = candidateBands.filter(
  ee.Filter.inList('item', stackBands)
);

var trainingPoolTable = allTrainingPoints
  .select(bandList.cat(['class_id']))
  .map(function(feature) {
    return feature.select(feature.propertyNames().removeAll(['system:index', '.geo']));
  });


// ---------------------------------------------------------------------
// 8. External evaluation table
// ---------------------------------------------------------------------
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


var rawEvaluationPoints = ee.FeatureCollection(evaluationPointsAsset)
  .filterBounds(studyArea);

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
  .filter(ee.Filter.inList('class_id', CLASS_IDS));

var evaluationTable = predictorStack
  .select(bandList)
  .sampleRegions({
    collection: evaluationPoints,
    properties: ['class_id'],
    scale: SCALE,
    geometries: false,
    tileScale: 4
  })
  .filter(ee.Filter.notNull(bandList))
  .map(function(feature) {
    return feature.select(feature.propertyNames().removeAll(['system:index', '.geo']));
  });


// ---------------------------------------------------------------------
// 9. Export tables
// ---------------------------------------------------------------------
Export.table.toDrive({
  collection: trainingPoolTable,
  description: 'TRAINPOOL_' + SETUP.name + '_' + YEAR,
  folder: DRIVE_FOLDER_MDA,
  fileNamePrefix: 'TRAINPOOL_' + SETUP.name + '_' + YEAR,
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: evaluationTable,
  description: 'EVAL_' + SETUP.name + '_' + YEAR,
  folder: DRIVE_FOLDER_MDA,
  fileNamePrefix: 'EVAL_' + SETUP.name + '_' + YEAR,
  fileFormat: 'CSV'
});


print('MDA input tables prepared. Run the export tasks from the Tasks tab.');