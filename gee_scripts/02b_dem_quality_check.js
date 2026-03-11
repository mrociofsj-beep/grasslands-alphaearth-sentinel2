/*
DEM-derived predictor stack aligned to a 10 m reference grid

This script:
1. Loads the Copernicus GLO-30 DEM
2. Resamples the DEM to a 10 m reference grid
3. Derives topographic predictors
4. Builds a final DEM predictor stack
5. Exports the stack to both Earth Engine Assets and Google Drive

Derived variables:
- elevation
- slope
- aspect_sin
- aspect_cos
- rugosity

Notes:
- Slope is derived using a Sobel operator on the 10 m resampled DEM
- Drive export uses -9999 as NoData
*/

var REF_CRS = 'EPSG:32721';
var REF_TRANSFORM = [10, 0, 147208.08762427027, 0, -10, 5952049.19864255];

var studyArea = ee.FeatureCollection('projects/your_project/assets/study_area')
  .geometry()
  .dissolve();


function clipToStudyArea(image) {
  return image.clip(studyArea).toFloat();
}


// ---------------------------------------------------------------------
// 1. Base DEM
// ---------------------------------------------------------------------
var demRaw = ee.ImageCollection('COPERNICUS/DEM/GLO30')
  .filterBounds(studyArea.buffer(2000))
  .mosaic()
  .select('DEM')
  .clip(studyArea.buffer(2000));

var demFilled = demRaw.unmask(0);


// ---------------------------------------------------------------------
// 2. Resample to 10 m reference grid
// ---------------------------------------------------------------------
var dem10m = demFilled
  .resample('bilinear')
  .reproject(REF_CRS, REF_TRANSFORM, null);


// ---------------------------------------------------------------------
// 3. Topographic derivatives
// ---------------------------------------------------------------------
var terrainProducts = ee.Terrain.products(dem10m);

var aspectRad = terrainProducts.select('aspect')
  .unmask(0)
  .multiply(Math.PI / 180);

var aspectSin = aspectRad.sin().rename('aspect_sin');
var aspectCos = aspectRad.cos().rename('aspect_cos');


// ---------------------------------------------------------------------
// 4. Rugosity (~90 m neighborhood)
// ---------------------------------------------------------------------
var rugosityKernel = ee.Kernel.circle({
  radius: 45,
  units: 'meters',
  normalize: false
});

var demMasked = dem10m.updateMask(dem10m.neq(0));

var localMax = demMasked.reduceNeighborhood({
  reducer: ee.Reducer.max(),
  kernel: rugosityKernel
});

var localMin = demMasked.reduceNeighborhood({
  reducer: ee.Reducer.min(),
  kernel: rugosityKernel
});

var rugosity = localMax.subtract(localMin)
  .rename('rugosity')
  .toFloat();


// ---------------------------------------------------------------------
// 5. Slope from Sobel operator
// ---------------------------------------------------------------------
var cellSize = 10;

var sobelX = ee.Kernel.fixed(
  3, 3,
  [[-1, 0, 1],
   [-2, 0, 2],
   [-1, 0, 1]],
  1, 1, false
);

var sobelY = ee.Kernel.fixed(
  3, 3,
  [[-1, -2, -1],
   [ 0,  0,  0],
   [ 1,  2,  1]],
  1, 1, false
);

var dzdx = dem10m.convolve(sobelX).divide(8 * cellSize);
var dzdy = dem10m.convolve(sobelY).divide(8 * cellSize);

var slope = dzdx.hypot(dzdy)
  .atan()
  .multiply(180 / Math.PI)
  .rename('slope')
  .toFloat();


// ---------------------------------------------------------------------
// 6. Final predictor stack
// ---------------------------------------------------------------------
var bandOrder = ['elev', 'slope', 'aspect_sin', 'aspect_cos', 'rugosity'];

var demStack = clipToStudyArea(
  ee.Image.cat([
    dem10m.rename('elev'),
    slope,
    aspectSin,
    aspectCos,
    rugosity
  ]).select(bandOrder)
);


// ---------------------------------------------------------------------
// 7. Export
// ---------------------------------------------------------------------
Export.image.toAsset({
  image: demStack,
  description: 'DEM_stack_10m',
  assetId: 'projects/your_project/assets/DEM_stack_10m',
  region: studyArea,
  crs: REF_CRS,
  crsTransform: REF_TRANSFORM,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: demStack.unmask(-9999),
  description: 'DEM_stack_10m_DRIVE',
  folder: 'dem_exports',
  fileNamePrefix: 'DEM_stack_10m',
  region: studyArea,
  crs: REF_CRS,
  crsTransform: REF_TRANSFORM,
  maxPixels: 1e13,
  formatOptions: {
    cloudOptimized: true,
    noData: -9999
  },
  skipEmptyTiles: false
});


print('DEM predictor stack prepared. Run the tasks from the Tasks tab.');