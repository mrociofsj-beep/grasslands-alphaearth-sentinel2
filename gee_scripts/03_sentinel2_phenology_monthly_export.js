/*
Sentinel-2 spectral and phenological predictor export

This script:
1. Loads Sentinel-2 surface reflectance and cloud probability data
2. Applies cloud masking and optional topographic correction
3. Builds a spectral composite for a user-defined time range
4. Derives phenological predictors from the same time range
5. Exports both products to Earth Engine Assets and Google Drive

Important:
- The input date range should span full months
- Monthly intervals are required to compute robust phenological metrics
- At least two monthly periods are needed to derive NDVI_slope

Exports:
- Sentinel-2 spectral predictor composite
- Phenological predictor stack
*/

var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var exportRegion = studyArea.bounds(1, ee.Projection(REF_CRS));


// ---------------------------------------------------------------------
// 1. Time range
// ---------------------------------------------------------------------
// The selected interval should cover complete months.
// Recommended format:
// START_DATE = first day of a month
// END_DATE   = last day of a month
// Example: 2023-10-01 to 2023-12-31
var START_DATE = '2023-10-01';
var END_DATE   = '2023-12-31';


// ---------------------------------------------------------------------
// 2. Export controls
// ---------------------------------------------------------------------
var EXPORT_S2 = true;
var EXPORT_PHENO = true;


// ---------------------------------------------------------------------
// 3. Cloud and scene filtering parameters
// ---------------------------------------------------------------------
var params = {
  MAX_CLOUD_PROBABILITY: 30,
  MAX_SCENE_CLOUD_COVER: 10,
  MAX_CLOUD_PERCENT_IN_AOI: 4
};


// ---------------------------------------------------------------------
// 4. Band order
// ---------------------------------------------------------------------
var S2_BAND_ORDER = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'];
var PHENO_BAND_ORDER = ['NDVI_med', 'EVI_med', 'NDVI_p10', 'NDVI_p90', 'NDVI_rng', 'NDVI_slope'];


// ---------------------------------------------------------------------
// 5. Helper functions
// ---------------------------------------------------------------------
function joinByIndex(primaryCollection, secondaryCollection) {
  return ee.Join.saveFirst('cloud_mask').apply({
    primary: primaryCollection,
    secondary: secondaryCollection,
    condition: ee.Filter.equals({
      leftField: 'system:index',
      rightField: 'system:index'
    })
  });
}


function hasRequiredBands(image) {
  var requiredBands = ['B2', 'B3', 'B4', 'B8'];
  var bandNames = image.bandNames();

  var allPresent = ee.List(requiredBands).iterate(function(bandName, acc) {
    return ee.Number(acc).multiply(bandNames.indexOf(bandName).neq(-1));
  }, 1);

  return image.set('bands_ok', allPresent);
}


function computeCloudPercentage(image, cloudCollection) {
  var imageId = image.get('system:index');

  var cloudProbability = ee.Image(
    cloudCollection
      .filter(ee.Filter.eq('system:index', imageId))
      .first()
  ).select('probability');

  var cloudMask = cloudProbability.gt(params.MAX_CLOUD_PROBABILITY);

  var stats = cloudMask.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: studyArea,
    scale: 20,
    maxPixels: 1e13,
    tileScale: 2,
    bestEffort: true
  });

  var fraction = ee.Number(stats.get('probability'));
  var percentage = ee.Algorithms.If(fraction, fraction.multiply(100), 100);

  return image.set('cloud_percentage_aoi', percentage);
}


// ---------------------------------------------------------------------
// 6. Cloud masking
// ---------------------------------------------------------------------
var NDVI_VEGETATION_THRESHOLD = 0.20;
var STRONG_CLOUD_THRESHOLD = 70;

function maskClouds(image) {
  var cloudImage = ee.Image(image.get('cloud_mask'));
  var probability = cloudImage.select('probability').unmask(0);
  var scl = image.select('SCL').unmask(0);

  var strongCloud = probability.gte(STRONG_CLOUD_THRESHOLD);
  var cloudBuffer = strongCloud.focal_max(30, 'square', 'meters');
  var probabilityMask = probability.lt(params.MAX_CLOUD_PROBABILITY).and(cloudBuffer.not());

  var invalidSCL = scl.remap(
    [0, 1, 2, 3, 7, 8, 9, 10, 11, 12],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    0
  );

  var baseValidMask = probabilityMask.and(invalidSCL.eq(0));

  var ndvi = image.normalizedDifference(['B8', 'B4']);
  var vegetationRecovery = ndvi.gte(NDVI_VEGETATION_THRESHOLD)
    .and(scl.neq(3))
    .and(probability.lt(STRONG_CLOUD_THRESHOLD));

  var finalMask = baseValidMask.or(vegetationRecovery).unmask(0);

  return image.updateMask(finalMask);
}


// ---------------------------------------------------------------------
// 7. Optional topographic correction
// ---------------------------------------------------------------------
var APPLY_TOPOGRAPHIC_CORRECTION = true;

var demRaw = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .filterBounds(studyArea.buffer(2000))
  .mosaic()
  .select(0)
  .rename('DEM')
  .clip(studyArea.buffer(2000));

var dem10m = demRaw.unmask(0).resample('bilinear');

var terrain = ee.Terrain.products(dem10m);
var slopeDeg = terrain.select('slope').unmask(0).rename('slope');
var aspectRad = terrain.select('aspect').unmask(0).multiply(Math.PI / 180);
var slopeRad = slopeDeg.multiply(Math.PI / 180);

var cosSlope = slopeRad.cos();
var sinSlope = slopeRad.sin();
var cosAspect = aspectRad.cos();
var sinAspect = aspectRad.sin();


function applyTopographicCorrection(image) {
  var validMask = image.select('B2').mask();

  var solarAzimuthDeg = ee.Algorithms.If(
    image.get('MEAN_SOLAR_AZIMUTH_ANGLE'),
    ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')),
    180
  );

  var solarZenithDeg = ee.Algorithms.If(
    image.get('MEAN_SOLAR_ZENITH_ANGLE'),
    ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE')),
    35
  );

  var solarAzimuth = ee.Number(solarAzimuthDeg).multiply(Math.PI / 180);
  var solarZenith = ee.Number(solarZenithDeg).multiply(Math.PI / 180);

  var cosSolarZenith = ee.Image.constant(solarZenith.cos());
  var sinSolarZenith = ee.Image.constant(solarZenith.sin());

  var cosAspectMinusAzimuth = cosAspect.multiply(solarAzimuth.cos())
    .add(sinAspect.multiply(solarAzimuth.sin()));

  var cosIncidence = cosSlope.multiply(cosSolarZenith)
    .add(sinSlope.multiply(sinSolarZenith).multiply(cosAspectMinusAzimuth))
    .max(0.05)
    .unmask(1);

  var correctionFactor = cosSolarZenith.divide(cosIncidence)
    .unmask(1)
    .rename('correction_factor');

  var opticalBands = image.select(['^B2$', '^B3$', '^B4$', '^B5$', '^B6$', '^B7$', '^B8$']);

  var correctedOptical = opticalBands
    .divide(10000)
    .multiply(correctionFactor)
    .clamp(0, 1)
    .multiply(10000)
    .updateMask(validMask);

  return image.addBands(correctedOptical, null, true);
}


// ---------------------------------------------------------------------
// 8. Prepare spectral bands
// ---------------------------------------------------------------------
function prepareSpectralBands(image) {
  var baseReflectance = image.select(['B2', 'B3', 'B4', 'B8'])
    .divide(10000)
    .float();

  var additionalReflectance = image.select(['B5', 'B6', 'B7'])
    .resample('bilinear')
    .divide(10000)
    .float();

  var output = ee.Image([])
    .addBands(baseReflectance)
    .addBands(additionalReflectance)
    .toFloat();

  return output
    .copyProperties(image, image.propertyNames())
    .set('system:time_start', image.get('system:time_start'));
}


// ---------------------------------------------------------------------
// 9. Build processed Sentinel-2 collection
// ---------------------------------------------------------------------
function buildProcessedCollection(startDate, endDate) {
  var s2SurfaceReflectance = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(studyArea)
    .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', params.MAX_SCENE_CLOUD_COVER));

  var s2CloudProbability = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    .filterDate(startDate, endDate)
    .filterBounds(studyArea)
    .filter(ee.Filter.inList('system:index', s2SurfaceReflectance.aggregate_array('system:index')));

  var s2Matched = s2SurfaceReflectance.filter(
    ee.Filter.inList('system:index', s2CloudProbability.aggregate_array('system:index'))
  );

  var s2Valid = ee.ImageCollection(s2Matched.map(hasRequiredBands))
    .filter(ee.Filter.eq('bands_ok', 1));

  var s2WithCloudStats = s2Valid.map(function(image) {
    return computeCloudPercentage(image, s2CloudProbability);
  });

  var s2Filtered = s2WithCloudStats.filter(
    ee.Filter.lte('cloud_percentage_aoi', params.MAX_CLOUD_PERCENT_IN_AOI)
  );

  var joined = joinByIndex(s2Filtered, s2CloudProbability);
  var cloudMasked = ee.ImageCollection(joined).map(maskClouds);

  var corrected = APPLY_TOPOGRAPHIC_CORRECTION
    ? cloudMasked.map(applyTopographicCorrection)
    : cloudMasked;

  return corrected.map(prepareSpectralBands);
}


// ---------------------------------------------------------------------
// 10. Build Sentinel-2 spectral composite
// ---------------------------------------------------------------------
function buildSpectralComposite(startDate, endDate) {
  var collection = buildProcessedCollection(startDate, endDate);

  var medianImage = collection.median()
    .clip(exportRegion)
    .toFloat();

  var validMask = medianImage.select(['B2', 'B3', 'B4', 'B8'])
    .mask()
    .reduce(ee.Reducer.min());

  return medianImage
    .updateMask(validMask)
    .clip(studyArea)
    .select(S2_BAND_ORDER)
    .toFloat();
}


// ---------------------------------------------------------------------
// 11. Add NDVI and EVI to the processed collection
// ---------------------------------------------------------------------
function addVegetationIndices(collection) {
  return collection.map(function(image) {
    var b8 = image.select('B8');
    var b4 = image.select('B4');
    var b2 = image.select('B2');

    var ndvi = b8.subtract(b4)
      .divide(b8.add(b4))
      .rename('NDVI');

    var numerator = b8.subtract(b4).multiply(2.5);
    var denominator = b8.add(b4.multiply(6))
      .subtract(b2.multiply(7.5))
      .add(1);

    denominator = denominator.where(denominator.abs().lt(1e-6), 1e-6);

    var evi = numerator.divide(denominator)
      .rename('EVI')
      .clamp(-1, 1);

    return image.addBands([ndvi.toFloat(), evi.toFloat()]);
  });
}


// ---------------------------------------------------------------------
// 12. Monthly NDVI medians within the selected range
// ---------------------------------------------------------------------
function monthlyNDVIWithinRange(startDate, endDate) {
  var collection = addVegetationIndices(buildProcessedCollection(startDate, endDate));

  var start = ee.Date(startDate);
  var end = ee.Date(endDate);

  var numberOfMonths = ee.Number(end.difference(start, 'month')).ceil().max(1);

  var monthlyImages = ee.List.sequence(0, numberOfMonths.subtract(1)).map(function(i) {
    var monthStart = start.advance(ee.Number(i), 'month');
    var monthEnd = monthStart.advance(1, 'month');
    var adjustedMonthEnd = ee.Date(ee.Number(monthEnd.millis()).min(end.millis()));

    var monthlyCollection = collection.filterDate(monthStart, adjustedMonthEnd).select('NDVI');
    var hasImages = monthlyCollection.size().gt(0);

    var emptyImage = ee.Image(0)
      .updateMask(ee.Image(0))
      .rename('NDVI');

    return ee.Image(ee.Algorithms.If(hasImages, monthlyCollection.median(), emptyImage))
      .set('system:time_start', monthStart.millis());
  });

  var monthlyCollection = ee.ImageCollection.fromImages(monthlyImages)
    .sort('system:time_start');

  return monthlyCollection.map(function(image) {
    var timeBand = ee.Image.constant(
      ee.Number(image.get('system:time_start'))
        .subtract(start.millis())
        .divide(1000 * 60 * 60 * 24)
    ).rename('t');

    return timeBand.addBands(image.select('NDVI')).toFloat();
  });
}


// ---------------------------------------------------------------------
// 13. Build phenological predictor stack
// ---------------------------------------------------------------------
function buildPhenologyComposite(startDate, endDate) {
  var collection = addVegetationIndices(buildProcessedCollection(startDate, endDate));

  var ndviMedian = collection.select('NDVI').median().rename('NDVI_med');
  var eviMedian = collection.select('EVI').median().rename('EVI_med');

  var ndviPercentiles = collection.select('NDVI').reduce(ee.Reducer.percentile([10, 90]));
  var ndviP10 = ndviPercentiles.select('NDVI_p10');
  var ndviP90 = ndviPercentiles.select('NDVI_p90');
  var ndviRange = ndviP90.subtract(ndviP10).rename('NDVI_rng');

  var monthlyCollection = monthlyNDVIWithinRange(startDate, endDate);
  var monthCount = monthlyCollection.size();

  var ndviSlope = ee.Image(ee.Algorithms.If(
    monthCount.gte(2),
    monthlyCollection
      .map(function(image) {
        return image.select(['t', 'NDVI']);
      })
      .reduce(ee.Reducer.linearFit())
      .select('scale')
      .rename('NDVI_slope'),
    ee.Image(0).updateMask(ee.Image(0)).rename('NDVI_slope')
  ));

  var referenceCollection = buildProcessedCollection(startDate, endDate);
  var referenceMedian = referenceCollection.median()
    .clip(exportRegion)
    .toFloat();

  var validMask = referenceMedian.select(['B2', 'B3', 'B4', 'B8'])
    .mask()
    .reduce(ee.Reducer.min());

  return ee.Image.cat([
      ndviMedian,
      eviMedian,
      ndviP10,
      ndviP90,
      ndviRange,
      ndviSlope
    ])
    .updateMask(validMask)
    .clip(studyArea)
    .select(PHENO_BAND_ORDER)
    .toFloat();
}


// ---------------------------------------------------------------------
// 14. Export products
// ---------------------------------------------------------------------
var label = START_DATE + '_to_' + END_DATE;

if (EXPORT_S2) {
  var s2Composite = buildSpectralComposite(START_DATE, END_DATE);
  var s2Name = 'S2_' + label;

  Export.image.toAsset({
    image: s2Composite,
    description: s2Name,
    assetId: 'projects/your_project/assets/' + s2Name,
    region: exportRegion,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: s2Composite.unmask(-9999).toFloat(),
    description: s2Name + '_DRIVE',
    folder: 'sentinel2_exports',
    fileNamePrefix: s2Name,
    region: exportRegion,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    },
    skipEmptyTiles: false
  });
}

if (EXPORT_PHENO) {
  var phenologyComposite = buildPhenologyComposite(START_DATE, END_DATE);
  var phenoName = 'PHENO_' + label;

  Export.image.toAsset({
    image: phenologyComposite,
    description: phenoName,
    assetId: 'projects/your_project/assets/' + phenoName,
    region: exportRegion,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: phenologyComposite.unmask(-9999).toFloat(),
    description: phenoName + '_DRIVE',
    folder: 'phenology_exports',
    fileNamePrefix: phenoName,
    region: exportRegion,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13,
    formatOptions: {
      cloudOptimized: true,
      noData: -9999
    },
    skipEmptyTiles: false
  });
}


print('Sentinel-2 and phenological exports prepared. Run the tasks from the Tasks tab.');