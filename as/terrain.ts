import { mandelbrot } from  "./mandelbrot.ts";

const CUBE_SIZE: i32 = 4;
const SECTOR_SIZE: i32 = 16;
const CUBE_STRIDE: i32 = 32 * 4;
const OBJECT_STRIDE: i32 = 1;
const SECTOR_LINES: i32 = 32;

/*
function cellularNoise2D(px, py, size)
{
    const cubeSize = 1.0 / size;

    // in which section am I?
    const qx = Math.floor(px / cubeSize);
    const qy = Math.floor(px / cubeSize);

    // check the surroundings
    let minDist = 9999.0;
    for (let x = -1; x < 2; ++x)
    {
        for (let y = -1; y < 2; ++y)
        {
            const sampleCubeX = qx + x;
            const sampleCubeY = qy + y;

            const moduloPointX = (sampleCubeX + size) / size;
            const moduloPointY = (sampleCubeY + size) / size;

            const randomPointX = Math.random(moduloPointX) + Math.sin(moduloPointX) * 0.1;
            const randomPointY = Math.random(moduloPointY) + Math.sin(moduloPointY) * 0.2;

            const samplePointX = (sampleCubeX + randomPointX) * cubeSize;
            const samplePointY = (sampleCubeY + randomPointY) * cubeSize;

            const dist = Math.sqrt((samplePointX - px) * (samplePointX - px) + (samplePointY - py) * (samplePointY - py));
            minDist = Math.min(minDist, dist);
        }
    }
    return minDist / cubeSize;
}
*/

function addObject(data: Int32Array, type: i32, x: i32, y: i32, z: i32, lod: i32): void
{
    const cubeLod: i32 = min(lod, 2);
    const sectorLod: i32 = max(0, lod - 2);
    const cubeSize: i32 = CUBE_SIZE / (1 << cubeLod);
    const sectorSize: i32 = SECTOR_SIZE / (1 << sectorLod);
    const bitsPerCoord: i32 = 2 / (1 << cubeLod);

    const cubeX: i32 = x / cubeSize;
    const cubeY: i32 = y / cubeSize;
    const cubeZ: i32 = z / cubeSize;

    const cubeIndex = cubeX * sectorSize * sectorSize + cubeY * sectorSize + cubeZ;

    const cubeDataOffset = cubeIndex * CUBE_STRIDE;

    let patternHi: u16 = <u16> data[cubeDataOffset];
    let patternHiMid: u16 = <u16> data[cubeDataOffset + 1];
    let patternLoMid: u16 = <u16> data[cubeDataOffset + 2];
    let patternLo: u16 = <u16> data[cubeDataOffset + 3];

    const objIndex = ((x - cubeX * cubeSize) << (bitsPerCoord + bitsPerCoord)) +
                     ((y - cubeY * cubeSize) << bitsPerCoord) +
                     (z - cubeZ * cubeSize);
    const objDataOffset = cubeDataOffset + 2 * 4 + objIndex * OBJECT_STRIDE;

    if (objIndex < 16)
    {
        patternLo |= 1 << <u16> objIndex;
    }
    else if (objIndex < 32)
    {
        patternLoMid |= 1 << (<u16> objIndex - 16);
    }
    else if (objIndex < 48)
    {
        patternHiMid |= 1 << (<u16> objIndex - 32);
    }
    else
    {
        patternHi |= 1 << (<u16> objIndex - 48);
    }

    data[cubeDataOffset] = patternHi;
    data[cubeDataOffset + 1] = patternHiMid;
    data[cubeDataOffset + 2] = patternLoMid;
    data[cubeDataOffset + 3] = patternLo;

    data[objDataOffset] = type;
}

export function generateSector(ux: i32, uy: i32, uz: i32, lod: i32): Int32Array
{
    // a sector consists of SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE cubes

    const cubeLod: i32 = min(lod, 2);
    const sectorLod: i32 = max(0, lod - 2);
    const cubeSize: i32 = CUBE_SIZE / (1 << cubeLod);
    const sectorSize: i32 = SECTOR_SIZE / (1 << sectorLod);

    const data = new Int32Array(SECTOR_LINES * 4096 * 4); //SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE * CUBE_STRIDE);
    data.fill(0, 0);

    const fullResScale = SECTOR_SIZE * CUBE_SIZE;
    const lodScale = sectorSize * cubeSize;

    for (let col: i32 = 0; col < lodScale; ++col)
    {
        for (let row: i32 = 0; row < lodScale; ++row)
        {
            const worldX = ux * fullResScale + col * (1 << lod);
            const worldZ = uz * fullResScale + row * (1 << lod);

            const px: f64 = <f64>(worldX - (-40 * fullResScale)) / <f64>(40 * fullResScale - -40 * fullResScale);
            const py: f64 = <f64>(worldZ - (-40 * fullResScale)) / <f64>(40 * fullResScale - -40 * fullResScale);

            const mx = -1.0 + px * 2.0;
            const my = -1.0 + py * 2.0;
            const height = 65 - mandelbrot(mx, my, 64);

            for (let layer: i32 = 0; layer < lodScale; ++layer)
            {
                const worldY = uy * fullResScale + layer * (1 << lod);

                if (worldY < height)
                {
                    let type: i32 = 2; // grass
                    if (worldY < 8)
                    {
                        type = 3; // sand
                    }
                    else if (worldY < height - 2)
                    {
                        type = 4; // rocks
                    }
                    addObject(data, type, col, layer, row, lod);
                }

                // fill valleys with water
                if (worldY < 2)
                {
                    addObject(data, 1, col, layer, row, lod);
                }
            }
        }
    }

    return data;
}
