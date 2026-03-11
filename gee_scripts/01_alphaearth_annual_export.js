/*
Annual AlphaEarth embedding export aligned to a reference grid

This script:
1. Loads the annual AlphaEarth embedding collection
2. Builds the annual composite for 2023
3. Aligns the image to a predefined reference grid
4. Exports the output to both Earth Engine Assets and Google Drive

Notes:
- Asset export preserves the mask
- Drive export uses -9999 as NoData
*/

var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();

var exportRegion = studyArea.bounds(1, ee.Projection(REF_CRS));

var year = 2023;
var alphaEarthCollection = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL');


function getAnnualEmbedding(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');

  return alphaEarthCollection
    .filterDate(start, end)
    .filterBounds(studyArea)
    .median()
    .toFloat();
}


function alignToReferenceGrid(image) {
  return image
    .clip(studyArea)
    .reproject(REF_CRS, REF_TRANSFORM, null)
    .toFloat();
}


function exportAnnualEmbedding(year) {
  var annualImage = getAnnualEmbedding(year);
  var alignedImage = alignToReferenceGrid(annualImage);
  var exportName = 'AE_' + year;

  Export.image.toAsset({
    image: alignedImage,
    description: exportName,
    assetId: 'projects/your_project/assets/' + exportName,
    region: exportRegion,
    crs: REF_CRS,
    crsTransform: REF_TRANSFORM,
    maxPixels: 1e13
  });

  Export.image.toDrive({
    image: alignedImage.unmask(-9999),
    description: exportName + '_DRIVE',
    folder: 'alphaearth_annual_exports',
    fileNamePrefix: exportName,
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


exportAnnualEmbedding(year);

print('Annual AlphaEarth export prepared. Run the tasks from the Tasks tab.');