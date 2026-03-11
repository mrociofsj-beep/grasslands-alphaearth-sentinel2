/*
Reference point cleaning and filtering for machine learning

This script:
1. Loads raw reference points
2. Standardizes class labels and maps them to integer class IDs
3. Removes points with missing or invalid classes
4. Keeps one point per 10 m pixel per class
5. Optionally applies a minimum spacing filter by class
6. Optionally limits the number of points per class
7. Exports the cleaned point dataset
8. Exports an additional table for spatial autocorrelation analyses

Notes:
- The class mapping includes only the target classes used in the study
- The reference grid matches the spatial grid used for predictor layers
*/

var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var rawPointsAsset = 'projects/your_project/assets/raw_reference_points';
var classProperty = 'class';


// ---------------------------------------------------------------------
// 1. Optional sampling filters
// ---------------------------------------------------------------------
var APPLY_MINIMUM_SPACING = true;
var MINIMUM_SPACING_METERS = 50;

var APPLY_CLASS_CAP = false;
var MAX_POINTS_PER_CLASS = 15000;


// ---------------------------------------------------------------------
// 2. Class mapping
// ---------------------------------------------------------------------
var CLASS_MAP = ee.Dictionary({
  'rocky_grassland': 1,
  'grassland': 2,
  'cropland': 3,
  'water': 4,
  'urban': 5,
  'forest_plantation': 6
});


// ---------------------------------------------------------------------
// 3. Load and standardize points
// ---------------------------------------------------------------------
var rawPoints = ee.FeatureCollection(rawPointsAsset)
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

  return classValue;
}


var taggedPoints = rawPoints.map(function(feature) {
  var hasClass = feature.propertyNames().contains(classProperty);

  var classLabel = ee.String(
    ee.Algorithms.If(hasClass, feature.get(classProperty), '')
  );

  classLabel = normalizeClassLabel(classLabel);

  var classId = ee.Number(
    ee.Algorithms.If(
      CLASS_MAP.contains(classLabel),
      CLASS_MAP.get(classLabel),
      -99
    )
  );

  return feature.set({
    class_label: classLabel,
    class_id: classId,
    has_class: hasClass
  });
});


// ---------------------------------------------------------------------
// 4. Keep only valid classes
// ---------------------------------------------------------------------
var allowedClassLabels = CLASS_MAP.keys();

var validPoints = taggedPoints
  .filter(ee.Filter.neq('class_id', -99))
  .filter(ee.Filter.inList('class_label', allowedClassLabels));


// ---------------------------------------------------------------------
// 5. Reproject geometries to the reference CRS
// ---------------------------------------------------------------------
function transformToReferenceCRS(feature) {
  return feature.setGeometry(feature.geometry().transform(REF_CRS, 1));
}

var pointsInReferenceCRS = validPoints.map(transformToReferenceCRS);


// ---------------------------------------------------------------------
// 6. Keep one point per 10 m pixel per class
// ---------------------------------------------------------------------
var X0 = ee.Number(REF_TRANSFORM[2]);
var Y0 = ee.Number(REF_TRANSFORM[5]);
var PIXEL_SIZE = ee.Number(REF_TRANSFORM[0]);


function addGridId(feature, cellSize, fieldName) {
  var coordinates = feature.geometry().coordinates();
  var x = ee.Number(ee.List(coordinates).get(0));
  var y = ee.Number(ee.List(coordinates).get(1));

  var col = x.subtract(X0).divide(cellSize).floor().toInt64();
  var row = Y0.subtract(y).divide(cellSize).floor().toInt64();

  var gridId = col.multiply(1000000000).add(row);

  return feature.set(fieldName, gridId);
}


var pointsWithPixelId = pointsInReferenceCRS.map(function(feature) {
  return addGridId(feature, PIXEL_SIZE, 'pixel_id_10m');
});

var uniquePointsPerPixel = pointsWithPixelId.distinct(['class_id', 'pixel_id_10m']);


// ---------------------------------------------------------------------
// 7. Optional minimum spacing by class
// ---------------------------------------------------------------------
var spacedPoints = ee.FeatureCollection(
  ee.Algorithms.If(
    APPLY_MINIMUM_SPACING,
    uniquePointsPerPixel
      .map(function(feature) {
        return addGridId(feature, MINIMUM_SPACING_METERS, 'spacing_cell_id');
      })
      .distinct(['class_id', 'spacing_cell_id']),
    uniquePointsPerPixel
  )
);


// ---------------------------------------------------------------------
// 8. Optional class cap
// ---------------------------------------------------------------------
function limitPointsPerClass(featureCollection, maxPointsPerClass) {
  var classIds = ee.List(featureCollection.aggregate_array('class_id').distinct());

  var limitedCollections = classIds.map(function(classId) {
    classId = ee.Number(classId);

    return featureCollection
      .filter(ee.Filter.eq('class_id', classId))
      .randomColumn('random_value', 2025)
      .sort('random_value')
      .limit(maxPointsPerClass);
  });

  return ee.FeatureCollection(limitedCollections).flatten();
}


var finalPoints = ee.FeatureCollection(
  ee.Algorithms.If(
    APPLY_CLASS_CAP,
    limitPointsPerClass(spacedPoints, MAX_POINTS_PER_CLASS),
    spacedPoints
  )
);


// ---------------------------------------------------------------------
// 9. Export cleaned reference points
// ---------------------------------------------------------------------
Export.table.toAsset({
  collection: finalPoints,
  description: 'clean_reference_points',
  assetId: 'projects/your_project/assets/clean_reference_points'
});

Export.table.toDrive({
  collection: finalPoints,
  description: 'clean_reference_points_drive',
  folder: 'reference_points_exports',
  fileNamePrefix: 'clean_reference_points',
  fileFormat: 'GeoJSON'
});


// ---------------------------------------------------------------------
// 10. Export points for spatial autocorrelation analyses
// ---------------------------------------------------------------------
var pointsForSpatialAnalysis = finalPoints.map(function(feature) {
  var coordinates = feature.geometry().coordinates();

  return feature.set({
    x: ee.List(coordinates).get(0),
    y: ee.List(coordinates).get(1),
    class_id: feature.get('class_id')
  });
});

Export.table.toDrive({
  collection: pointsForSpatialAnalysis,
  description: 'reference_points_spatial_analysis',
  folder: 'reference_points_exports',
  fileNamePrefix: 'reference_points_spatial_analysis',
  fileFormat: 'CSV',
  selectors: ['x', 'y', 'class_id', 'class_label', 'system:index']
});


print('Reference point exports prepared. Run the tasks from the Tasks tab.');