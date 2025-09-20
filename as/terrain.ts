import { mandelbrot } from  "./mandelbrot.ts";

const LOD_CUBE_SIZE: i32[] =   [ 4,  2,  1, 1, 1, 1, 1];
const LOD_SECTOR_SIZE: i32[] = [16, 16, 16, 8, 4, 2, 1];

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

function setVoxel(data: Uint32Array, type: i32, x: i32, y: i32, z: i32, lod: i32): void
{
    const cubeSize: i32 = LOD_CUBE_SIZE[lod];
    const sectorSize: i32 = LOD_SECTOR_SIZE[lod];
    const bitsPerCoord: i32 = lod == 0 ? 2 : 1;
    const cubeCount: i32 = sectorSize * sectorSize * sectorSize;
    const voxelCount: i32 = cubeSize * cubeSize * cubeSize;

    const cubeX: i32 = x / cubeSize;
    const cubeY: i32 = y / cubeSize;
    const cubeZ: i32 = z / cubeSize;

    const cubeIndex = cubeX * sectorSize * sectorSize +
                      cubeY * sectorSize +
                      cubeZ;

    const offset = cubeIndex * 4;

    let patternHi: u32 = <u32> data[offset];
    let patternLo: u32 = <u32> data[offset + 1];
    const address: i32 = cubeIndex;

    if (cubeSize > 1)
    {
        const voxelIndex = ((x - cubeX * cubeSize) << (bitsPerCoord + bitsPerCoord)) +
                           ((y - cubeY * cubeSize) << bitsPerCoord) +
                           (z - cubeZ * cubeSize);
        const objDataOffset = cubeCount * 4 + address * voxelCount + voxelIndex;

        if (voxelIndex < 32)
        {
            patternLo |= 1 << <u32> voxelIndex;
        }
        else
        {
            patternHi |= 1 << (<u32> voxelIndex - 32);
        }

        data[offset] = patternHi;
        data[offset + 1] = patternLo;
        data[offset + 2] = address;

        data[objDataOffset] = type;
    }
    else
    {
        const objDataOffset = cubeCount * 4 + address;

        patternLo |= 1;

        data[offset] = patternHi;
        data[offset + 1] = patternLo;
        data[offset + 2] = address;

        data[objDataOffset] = type;
    }
}

export function generateSector(ux: i32, uy: i32, uz: i32, lod: i32): Uint32Array
{
    // a sector consists of SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE cubes

    const cubeSize: i32 = LOD_CUBE_SIZE[lod];
    const sectorSize: i32 = LOD_SECTOR_SIZE[lod];

    const dataSize: i32 = sectorSize * sectorSize * sectorSize * 4 +
                          sectorSize * sectorSize * sectorSize * cubeSize * cubeSize * cubeSize;

    const data = new Uint32Array(dataSize);
    data.fill(0);

    const fullResScale = LOD_SECTOR_SIZE[0] * LOD_CUBE_SIZE[0];
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
                    setVoxel(data, type, col, layer, row, lod);
                }

                // fill valleys with water
                if (worldY < 4)
                {
                    setVoxel(data, 1, col, layer, row, lod);
                }
            }
        }
    }

    return data;
}
