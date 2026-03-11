/*
Training and test dataset preparation for multiple predictor setups

This script:
1. Loads the reference points and predictor stacks
2. Builds multi-temporal predictor stacks for Sentinel-2, DEM, phenology, and AlphaEarth
3. Filters reference points to retain only those with valid data across the full stack
4. Splits valid points into training and test subsets
5. Applies class balancing to the training set only
6. Exports training and test tables for multiple predictor setups
7. Exports the full valid point dataset for downstream analyses

Predictor setups:
- S2
- S2 + DEM
- S2 + PHENO
- S2 + DEM + PHENO
- AE
- AE + DEM
- AE + PHENO
- AE + DEM + PHENO

Notes:
- Class balancing is applied only to the training dataset
- The test dataset remains unbalanced
- The full valid point table is also exported for external analyses
*/

var YEAR = 2023;
var SCALE = 10;
var TEST_FRACTION = 0.30;
var RANDOM_SEED = 2025;
var MAX_TRAIN_POINTS_PER_CLASS = 500;
var NODATA = -9999;

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var referencePointsAsset = 'projects/your_project/assets/clean_reference_points';


// ---------------------------------------------------------------------
// 1. Predictor assets
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
// 2. Class definitions
// ---------------------------------------------------------------------
var classProperty = 'class';

var CLASS_MAP = ee.Dictionary({
  'rocky_grassland': 1,
  'grassland': 2,
  'cropland': 3,
  'water': 4,
  'urban': 5,
  'forest_plantation': 6
});

var allowedClassIds = [1, 2, 3, 4, 5, 6];

var idList = ee.List(CLASS_MAP.values());
var idStrings = idList.map(function(x) {
  return ee.String(x);
});

var idToName = ee.Dictionary.fromLists(idStrings, ee.List(CLASS_MAP.keys()));


function classNameFromId(id) {
  return ee.String(idToName.get(ee.String(ee.Number(id)), 'unknown'));
}


function maskNoData(image) {
  image = ee.Image(image);
  return image.updateMask(image.neq(NODATA));
}


function normalizeClassLabel(classValue) {
  classValue = ee.String(classValue)
    .trim()
    .toLowerCase()
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

  return classValue;
}


// ---------------------------------------------------------------------
// 3. Build predictor stacks
// ---------------------------------------------------------------------
function stackFromAssets(assetList, groupPrefix) {
  var images = assetList.map(function(assetId, index) {
    assetId = ee.String(assetId);
    index = ee.Number(index);

    var image = maskNoData(ee.Image(assetId));

    var prefix = ee.Algorithms.If(
      ee.String(groupPrefix).compareTo('DEM').eq(0)
        .or(ee.String(groupPrefix).compareTo('AE').eq(0)),
      ee.String(groupPrefix).cat('__'),
      ee.String(groupPrefix)
        .cat('_Q')
        .cat(index.add(1).format('%02d'))
        .cat('__')
    );

    var bandNames = image.bandNames();
    var renamedBands = bandNames.map(function(bandName) {
      return ee.String(prefix).cat(ee.String(bandName));
    });

    return image.rename(renamedBands);
  });

  var firstImage = ee.Image(ee.List(images).get(0));
  var remainingImages = ee.List(images).slice(1);

  return ee.Image(
    remainingImages.iterate(function(image, acc) {
      return ee.Image(acc).addBands(ee.Image(image), null, true);
    }, firstImage)
  );
}


function dropEmptyBands(image, region, scale) {
  var bandsToKeep = ee.List(
    image.bandNames().iterate(function(bandName, acc) {
      bandName = ee.String(bandName);
      acc = ee.List(acc);

      var count = image.select([bandName]).reduceRegion({
        reducer: ee.Reducer.count(),
        geometry: region,
        scale: scale,
        maxPixels: 1e13,
        tileScale: 2
      }).get(bandName);

      return ee.Algorithms.If(ee.Number(count).gt(0), acc.add(bandName), acc);
    }, ee.List([]))
  );

  return image.select(bandsToKeep);
}


var s2Stack = stackFromAssets(s2Assets, 'S2');
var demStack = stackFromAssets(demAssets, 'DEM');
var phenoStackRaw = stackFromAssets(phenoAssets, 'PH');
var phenoStack = dropEmptyBands(phenoStackRaw, studyArea, SCALE);
var aeStack = stackFromAssets(aeAssets, 'AE');

var s2Bands = s2Stack.bandNames();
var demBands = demStack.bandNames();
var phenoBands = phenoStack.bandNames();
var aeBands = aeStack.bandNames();

var s2DemBands = s2Bands.cat(demBands);
var s2PhenoBands = s2Bands.cat(phenoBands);
var s2AllBands = s2DemBands.cat(phenoBands);

var aeDemBands = aeBands.cat(demBands);
var aePhenoBands = aeBands.cat(phenoBands);
var aeAllBands = aeDemBands.cat(phenoBands);

var fullStack = s2Stack
  .addBands(demStack, null, true)
  .addBands(phenoStack, null, true)
  .addBands(aeStack, null, true);


// ---------------------------------------------------------------------
// 4. Load and filter reference points
// ---------------------------------------------------------------------
var rawPoints = ee.FeatureCollection(referencePointsAsset)
  .filterBounds(studyArea);

var referencePoints = rawPoints.map(function(feature) {
  var hasClassId = feature.propertyNames().contains('class_id');

  var classId = ee.Number(
    ee.Algorithms.If(
      hasClassId,
      feature.get('class_id'),
      ee.Algorithms.If(
        feature.propertyNames().contains(classProperty),
        ee.Algorithms.If(
          CLASS_MAP.contains(normalizeClassLabel(feature.get(classProperty))),
          CLASS_MAP.get(normalizeClassLabel(feature.get(classProperty))),
          -99
        ),
        -99
      )
    )
  );

  var classLabel = ee.String(
    ee.Algorithms.If(
      hasClassId,
      classNameFromId(classId),
      normalizeClassLabel(ee.String(feature.get(classProperty)))
    )
  );

  return feature.set({
    class_id: classId,
    class_label: classLabel
  });
})
.filter(ee.Filter.neq('class_id', -99))
.filter(ee.Filter.inList('class_id', allowedClassIds));


// ---------------------------------------------------------------------
// 5. Keep only points valid across the full predictor stack
// ---------------------------------------------------------------------
function buildFullMask(image) {
  return image.mask().reduce(ee.Reducer.min()).rename('valid_mask');
}

var validMask = buildFullMask(fullStack).clip(studyArea.simplify(50));
var stackProjection = fullStack.projection();

var pointsWithValidity = validMask.reduceRegions({
  collection: referencePoints,
  reducer: ee.Reducer.first(),
  scale: SCALE,
  crs: stackProjection,
  tileScale: 8
});

var validPoints = pointsWithValidity.filter(ee.Filter.eq('first', 1));


// ---------------------------------------------------------------------
// 6. Training/test split
// ---------------------------------------------------------------------
var pointsWithRandom = validPoints.randomColumn('rand_split', RANDOM_SEED);

var testPoints = pointsWithRandom.filter(ee.Filter.lt('rand_split', TEST_FRACTION));
var trainPoints = pointsWithRandom.filter(ee.Filter.gte('rand_split', TEST_FRACTION));


// ---------------------------------------------------------------------
// 7. Sample predictor values from the full stack
// ---------------------------------------------------------------------
function relaxMask(image) {
  var unionMask = image.mask().reduce(ee.Reducer.max());
  return image.updateMask(unionMask);
}

var sampledCollections = (function(image) {
  image = relaxMask(image);

  var trainSamples = image.sampleRegions({
    collection: trainPoints,
    properties: ['class_id'],
    scale: SCALE,
    geometries: true,
    tileScale: 8
  }).randomColumn('rand', RANDOM_SEED);

  var testSamples = image.sampleRegions({
    collection: testPoints,
    properties: ['class_id'],
    scale: SCALE,
    geometries: true,
    tileScale: 8
  }).randomColumn('rand', RANDOM_SEED);

  return {
    train: trainSamples,
    test: testSamples
  };
})(fullStack);

var trainFixed = ee.FeatureCollection(sampledCollections.train);
var testFixed = ee.FeatureCollection(sampledCollections.test);
var allFixed = trainFixed.merge(testFixed);


// ---------------------------------------------------------------------
// 8. Global training set balancing
// ---------------------------------------------------------------------
function limitPerClass(featureCollection, classField, limitsDict) {
  var classIds = ee.List(featureCollection.aggregate_array(classField).distinct());
  var emptyCollection = ee.FeatureCollection([]);

  return ee.FeatureCollection(
    classIds.iterate(function(classId, acc) {
      classId = ee.Number(classId);
      acc = ee.FeatureCollection(acc);

      var classLimit = ee.Number(ee.Dictionary(limitsDict).get(classId.format(), 0));

      var subset = featureCollection
        .filter(ee.Filter.eq(classField, classId))
        .limit(classLimit, 'rand', true);

      return acc.merge(subset);
    }, emptyCollection)
  );
}


function buildBalancedTrainingSet(rawTrainingSet, maxPointsPerClass) {
  var histogram = ee.Dictionary(rawTrainingSet.aggregate_histogram('class_id'));
  var counts = ee.List(histogram.values());

  var minimumPositiveCount = ee.Number(
    ee.Algorithms.If(
      counts.size().gt(0),
      ee.Number(
        ee.List(counts.sort()).iterate(function(value, acc) {
          acc = ee.Number(acc);
          value = ee.Number(value);

          return ee.Algorithms.If(
            acc.gt(0),
            acc,
            ee.Algorithms.If(value.gt(0), value, 0)
          );
        }, 0)
      ),
      0
    )
  );

  var maxRatio = 5;

  var classLimits = ee.Dictionary(
    ee.List(histogram.keys()).iterate(function(key, acc) {
      acc = ee.Dictionary(acc);

      var classCount = ee.Number(histogram.get(key));

      var classLimit = ee.Number(
        ee.Algorithms.If(
          minimumPositiveCount.gt(0),
          classCount.min(minimumPositiveCount.multiply(maxRatio)).min(maxPointsPerClass),
          classCount.min(maxPointsPerClass)
        )
      ).int();

      return acc.set(key, classLimit);
    }, ee.Dictionary({}))
  );

  return limitPerClass(rawTrainingSet, 'class_id', classLimits);
}


var balancedTrainingMaster = buildBalancedTrainingSet(
  trainFixed,
  MAX_TRAIN_POINTS_PER_CLASS
);


// ---------------------------------------------------------------------
// 9. Predictor setups
// ---------------------------------------------------------------------
var predictorSetups = [
  {name: 'S2', bands: s2Bands},
  {name: 'S2_DEM', bands: s2DemBands},
  {name: 'S2_PHENO', bands: s2PhenoBands},
  {name: 'S2_DEM_PHENO', bands: s2AllBands},
  {name: 'AE', bands: aeBands},
  {name: 'AE_DEM', bands: aeDemBands},
  {name: 'AE_PHENO', bands: aePhenoBands},
  {name: 'AE_DEM_PHENO', bands: aeAllBands}
];


// ---------------------------------------------------------------------
// 10. Export training and test tables for each setup
// ---------------------------------------------------------------------
predictorSetups.forEach(function(setup) {
  var setupName = setup.name;
  var selectedBands = ee.List(setup.bands);

  var trainingSet = balancedTrainingMaster.select(
    selectedBands.cat(['class_id', 'rand'])
  );

  var testSet = testFixed.select(
    selectedBands.cat(['class_id', 'rand'])
  );

  var trainingName = 'TRAIN_BALANCED_' + YEAR + '_' + setupName;
  var testName = 'TEST_RAW_' + YEAR + '_' + setupName;

  Export.table.toAsset({
    collection: trainingSet,
    description: trainingName,
    assetId: 'projects/your_project/assets/' + trainingName
  });

  Export.table.toAsset({
    collection: testSet,
    description: testName,
    assetId: 'projects/your_project/assets/' + testName
  });
});


// ---------------------------------------------------------------------
// 11. Export full valid point table
// ---------------------------------------------------------------------
Export.table.toAsset({
  collection: allFixed,
  description: 'TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK',
  assetId: 'projects/your_project/assets/TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK'
});

var propertiesToKeep = allFixed.first().propertyNames().removeAll([
  'system:index',
  'rand',
  '.geo'
]);

var allFixedClean = allFixed.select(propertiesToKeep);

Export.table.toDrive({
  collection: allFixedClean,
  description: 'TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK_csv',
  folder: 'training_dataset_exports',
  fileNamePrefix: 'TOTAL_VALID_POINTS_' + YEAR + '_FULL_STACK',
  fileFormat: 'CSV'
});


print('Training and test dataset exports prepared. Run the tasks from the Tasks tab.');