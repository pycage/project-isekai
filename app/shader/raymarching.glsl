#version 300 es
precision mediump float;
precision highp int;    /* Android needs this for 32 bit precision */
precision highp isampler2D;

// world configuration
const int horizonSize = 5;
const int sectorSize = 16;
const int cubeSize = 4;
const int cubeDataStride = 32;
const int worldPageSize = 4096;
const int sectorLines = 32;

in vec2 uv;
out vec4 fragColor;

uniform int timems;
uniform vec3 universeLocation;

uniform int marchingDepth;
uniform int tracingDepth;

uniform float fogDensity;
uniform bool enableShadows;
uniform bool enableAmbientOcclusion;
uniform bool enableToonEffect;
uniform bool enableTasm;
uniform bool showNormals;

uniform mat4 cameraTrafo;

uniform float screenWidth;
uniform float screenHeight;

uniform int numLights;
uniform isampler2D worldData;
uniform sampler2D lightsData;
uniform sampler2D tasmData;

struct CubeLocator
{
    int x;
    int y;
    int z;
    int sector;
};

struct ObjectLocator
{
    int x;
    int y;
    int z;
};

struct WorldLocator
{
    CubeLocator cube;
    ObjectLocator object;
};

struct ObjectAndDistance
{
    WorldLocator object;
    float distance;
};

float depthMap = 0.0;
int cubeDistanceMap = 0;
float cubeEntryDistanceMap = 0.0;
float voxelDistanceMap = 0.0;

bool freeEdge = false;
bool aoEdge = false;
int debug = 0;
vec3 debugColor = vec3(1.0, 0.0, 0.0);

float randomSeed = 0.0;


float[64] tasmRegisters;
const int REG_VOID = 0;
const int REG_PTR_VOID = 1;
const int REG_PC = 2;
const int REG_PTR_PC = 3;
const int REG_SP = 4;
const int REG_PTR_SP = 5;
const int REG_PARAM1 = 6;
const int REG_PARAM2 = 7;
const int REG_PTR_PARAM1 = 8;
const int REG_PTR_PARAM2 = 9;
const int REG_COLOR_R = 10;
const int REG_COLOR_G = 11;
const int REG_COLOR_B = 12;
const int REG_NORMAL_X = 13;
const int REG_NORMAL_Y = 14;
const int REG_NORMAL_Z = 15;
const int REG_ATTRIB_1 = 16;
const int REG_ATTRIB_2 = 17;
const int REG_ATTRIB_3 = 18;
const int REG_PTR_COLOR = 19;
const int REG_PTR_NORMAL = 20;
const int REG_PTR_ATTRIBUTES = 21;
const int REG_PTR_ATTRIB_2 = 22;
const int REG_PTR_ATTRIB_3 = 23;
const int REG_END_VALUE = 24;
const int REG_PTR_END_VALUE = 25;
const int REG_ENV_TIMEMS = 26;
const int REG_ENV_ST_X = 27;
const int REG_ENV_ST_Y = 28;
const int REG_ENV_RAY_DISTANCE = 29;
const int REG_ENV_P_X = 30;
const int REG_ENV_P_Y = 31;
const int REG_ENV_P_Z = 32;
const int REG_ENV_UNIVERSE_X = 33;
const int REG_ENV_UNIVERSE_Y = 34;
const int REG_ENV_UNIVERSE_Z = 35;
const int REG_STACK = 36;
const int REG_USER = 56;



int intr(float v)
{
    return int(round(v));
}

int mapSector(int sector)
{
    return texelFetch(worldData, ivec2(sector / 4, worldPageSize - 1), 0)[sector % 4];
}

ivec3 sectorLocation(int sector)
{
    int y = sector / (horizonSize * horizonSize);
    int z = (sector % (horizonSize * horizonSize)) / horizonSize;
    int x = sector % horizonSize;

    return ivec3(x, y, z);
}

int lodOfSector(int sector)
{
    ivec3 v = sectorLocation(sector);
    int center = horizonSize / 2;
    int dist = max(max(abs(v.x - center), abs(v.y - center)), abs(v.z - center));

    return dist < 2 ? 0 : 1;
}

int getCubeLod(int lod)
{
    return min(lod, 2);
}

int getSectorLod(int lod)
{
    return max(0, lod - 2);
}

vec3 resolveCubeLocator(CubeLocator cube)
{
    return vec3(sectorLocation(cube.sector)) * float(sectorSize * cubeSize) + vec3(
        float(cube.x * cubeSize),
        float(cube.y * cubeSize),
        float(cube.z * cubeSize)
    );
}

CubeLocator makeSuperCubeLocator(vec3 v, int level)
{
    v = clamp(v, 0.0, float(sectorSize * horizonSize * cubeSize - 1));
    int superCubeSize = cubeSize << level;
    int sectorLength = sectorSize * superCubeSize;

    ivec3 sectorLoc = ivec3(v) / sectorLength;
    int sector = sectorLoc.y * (horizonSize * horizonSize) + sectorLoc.z * horizonSize + sectorLoc.x;

    vec3 t = (v - vec3(sectorLoc * sectorLength)) / float(superCubeSize);

    return CubeLocator(int(t.x), int(t.y), int(t.z), sector);
}

bool isSuperCubeEmpty(CubeLocator superCube, int level)
{
    int index = superCube.x * sectorSize * sectorSize + superCube.y * sectorSize + superCube.z;
    ivec4 data = texelFetch(worldData, ivec2(index / 4, 3000 + level), 0);
    return data[index % 4] == 0;
}

ObjectLocator makeObjectLocator(vec3 locInCube)
{
    //locInCube = clamp(locInCube, 0.0, 4.0);
    int ox = int(floor(locInCube.x));
    int oy = int(floor(locInCube.y));
    int oz = int(floor(locInCube.z));
    return ObjectLocator(ox, oy, oz);
}

vec3 resolveObjectLocator(ObjectLocator obj)
{
    return vec3(
        float(obj.x) + 0.5,
        float(obj.y) + 0.5,
        float(obj.z) + 0.5
    );
}

WorldLocator makeWorldLocator(CubeLocator cube, ObjectLocator obj)
{
    return WorldLocator(cube, obj);
}

mat4 cubeTrafo(CubeLocator cube)
{
    vec3 p = resolveCubeLocator(cube);
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(p, 1.0)
    );
}

mat4 cubeTrafoInverse(CubeLocator cube)
{
    vec3 p = resolveCubeLocator(cube);
    return mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(-p, 1.0)
    );
}

ivec2 cubeDataOffset(CubeLocator cube)
{
    int sectorLod = getSectorLod(lodOfSector(cube.sector));
    int sectorSizeLod = sectorSize / (1 << sectorLod);

    // read sector map
    int sectorLine = sectorLines * mapSector(cube.sector);

    ivec3 cubeLoc = ivec3(cube.x, cube.y, cube.z);
    cubeLoc /= (1 << sectorLod);

    int index = cubeLoc.x * sectorSizeLod * sectorSizeLod + cubeLoc.y * sectorSizeLod + cubeLoc.z;
    int cubesPerLine = worldPageSize / cubeDataStride;
    return ivec2(
        (index % cubesPerLine) * cubeDataStride,
        sectorLine + index / cubesPerLine
    );
}

int objectDataEntry(WorldLocator worldLoc)
{
    int cubeLod = getCubeLod(lodOfSector(worldLoc.cube.sector));
    int bitsPerCoord = 2 / (1 << cubeLod);
    ivec3 loc = ivec3(worldLoc.object.x, worldLoc.object.y, worldLoc.object.z);
    loc /= (1 << cubeLod);
    int objLoc = (loc.x << (bitsPerCoord + bitsPerCoord)) + (loc.y << bitsPerCoord) + loc.z;
    ivec2 offset = cubeDataOffset(worldLoc.cube) + ivec2(2 + objLoc / 4, 0);
    return texelFetch(worldData, offset, 0)[objLoc % 4];
}

vec2 aabbMinMax(float origin, float dir, float boxMin, float boxMax)
{
    if (abs(dir) < 0.000001)
    {
        dir += 0.000001;
    }
    float tMin = (boxMin - origin) / dir;
    float tMax = (boxMax - origin) / dir;

    if (tMax < tMin)
    {
        return vec2(tMax, tMin);
    }
    else
    {
        return vec2(tMin, tMax);
    }
}

vec3 hitAabb(vec3 origin, vec3 rayDirection)
{
    vec2 tx = aabbMinMax(origin.x, rayDirection.x, -0.5, 0.5);
    vec2 ty = aabbMinMax(origin.y, rayDirection.y, -0.5, 0.5);
    vec2 tz = aabbMinMax(origin.z, rayDirection.z, -0.5, 0.5);

    // greatest min and smallest max
    float rayMin = (tx.s > ty.s) ? tx.s : ty.s;
    float rayMax = (tx.t < ty.t) ? tx.t : ty.t;

    if (tx.s > ty.t || ty.s > tx.t)
    {
        return vec3(0.0);
    }
    if (rayMin > tz.t || tz.s > rayMax)
    {
        return vec3(0.0);
    }
    if (tz.s > rayMin)
    {
        rayMin = tz.s;
    }
    if (tz.t < rayMax)
    {
        rayMax = tz.t;
    }
    return vec3(rayMin, rayMax, 1.0);
}

float lerp(float a, float b, float c)
{
    return a + (b - a) * c;
}

float seededRandom(vec2 st)
{
    st += vec2(randomSeed);
    randomSeed += 1.0;
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

mat2 rotate2d(float angle)
{
    return mat2(cos(angle), -sin(angle),
                sin(angle), cos(angle));
}

/* Creates a transformation matrix for transforming to surface space.
 */
mat4 createSurfaceTrafo(vec3 normal)
{
    vec3 p = vec3(1.0, 0.0, 0.0);

    vec3 tangent = p;

    // rotate normal around y counter-clockwise for second point
    if (abs(dot(normal, vec3(0.0, 1.0, 0.0))) < 1.0)
    {
        p = vec3(
            normal.x * cos(0.1) - normal.z * sin(0.1),
            0.0,
            normal.x * sin(0.1) - normal.z * cos(0.1)
        );

        float dp = dot(normal, p);
        tangent = normalize(p - normal * dp);
    }

    vec3 bitangent = normalize(cross(normal, tangent));
   
    return mat4(
        vec4(tangent, 0.0),
        vec4(bitangent, 0.0),
        vec4(normal, 0.0),
        vec4(0.0, 0.0, 0.0, 1.0)
    );
}

/*
vec2 mosaic(vec2 st, float size)
{
    return vec2(ceil(st.x * size) / size, ceil(st.y * size) / size);
}
*/

/* Generates a normal map from a height map.
 */
vec3 generateBumpNormal(float a, float b, float c, float d)
{
    return normalize(vec3(
        a - b,
        c - d,
        1.0
    ));
}

vec2 generateMipMap(vec2 st, int level)
{
    return floor(st * float(level)) / float(level);
}

float generateLine(vec2 st, float start, float end)
{
    return step(st.y, end) * (1.0 - step(st.y, start));
}

float generateWhiteNoise(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float generateCellularNoise2D(vec2 p, int size, float variant)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    // in which section am I?
    ivec2 q = ivec2(
        int(floor(p.x / cubeSize)),
        int(floor(p.y / cubeSize))
    );

    // check the surroundings
    float minDist = 9999.0;
    for (int x = -1; x < 2; ++x)
    {
        for (int y = -1; y < 2; ++y)
        {
            ivec2 sampleCube = q + ivec2(x, y);

            vec2 moduloPoint = vec2(
                float((sampleCube.x + size) % size),
                float((sampleCube.y + size) % size)
            );
            vec2 randomPoint = vec2(
                random(moduloPoint.xy) + sin(moduloPoint.x + variant) * 0.1,
                random(moduloPoint.yx * 0.1) + cos(moduloPoint.y + variant) * 0.2
            );
            vec2 samplePoint = (vec2(sampleCube) + randomPoint) * cubeSize;

            minDist = min(distance(samplePoint, p), minDist);
        }
    }
    return minDist / cubeSize;
}

float generateCellularNoise3D(vec3 p, int size)
{
    float fsize = float(size);
    float cubeSize = 1.0 / fsize;

    // in which section am I?
    ivec3 q = ivec3(
        int(floor(p.x / cubeSize)),
        int(floor(p.y / cubeSize)),
        int(floor(p.z / cubeSize))
    );

    // check the surroundings
    float minDist = 9999.0;
    for (int x = -1; x < 2; ++x)
    {
        for (int y = -1; y < 2; ++y)
        {
            for (int z = -1; z < 2; ++z)
            {
                vec3 sampleCube = vec3(q + ivec3(x, y, z));

                vec3 randomPoint = vec3(
                    random(sampleCube.xy),
                    random(sampleCube.xz),
                    random(sampleCube.yz)
                );
                vec3 samplePoint = (sampleCube + randomPoint) * cubeSize;

                minDist = min(distance(samplePoint, p), minDist);

            }
        }
    }
    return minDist / cubeSize;
}

float generateCheckerboard(vec2 st)
{
    float value1 = step(fract(st.x), 0.5);
    float value2 = step(fract(st.y), 0.5);
    return min(value1, value2) + (1.0 - max(value1, value2));
}

/*
float generateSteppedSin(vec2 st, float steps)
{
    return floor(steps * sin(st.x * 3.14195)) / steps;
}

float generateSteppedPyramid(vec2 st, float steps)
{
    float value1 = generateSteppedSin(st, steps);
    float value2 = generateSteppedSin(st.yx, steps);
    return min(value1, value2);
}

float generateTriangle(vec2 st)
{
    return step(clamp(st.x - st.y, 0.0, 1.0), 0.0);
}

float generateRipple(vec2 st, float p)
{
    return 0.5 + sin(
        pow(
            pow(abs(st.s), p) + pow(abs(st.t), p),
            (1.0 / p)
        )
    ) / 2.0;
}

float generateWaves(vec2 st)
{
    float e = 2.7183;
    return (pow(e, sin(st.s) * cos(st.t)) / (e * e));
}
*/



/* Procedural bricks texture.
 */
/*
mat3 pmatBricks(vec2 st)
{
    st = wrapSt(st);
    float value1 = generateLine(st, 0.0, 0.025);
    float value2 = generateLine(st, 0.975, 1.0);
    float value3 = generateLine(st, 0.47, 0.52);

    float value4 = generateLine(st.yx, 0.2, 0.25) * generateLine(st, 0.0, 0.47);
    float value5 = generateLine(st.yx, 0.65, 0.7) * generateLine(st, 0.52, 1.0);

    float linesMask = min(1.0, value1 + value2 + value3 + value4 + value5);

    float noise1 = 0.8 + generateWhiteNoise(st) * 0.2;
    float noise2 = 0.9 + generateWhiteNoise(st) * 0.1;

    vec3 color1 = vec3(0.95, 0.95, 0.77) * noise1;
    vec3 color2 = vec3(0.67, 0.44, 0.44); // * noise2;
    vec3 color = linesMask > 0.0 ? color1 : color2;

    float height = 3.0 * (1.0 - linesMask) * noise2;

    return mat3(color, vec3(height), vec3(1.0, 0.0, 0.0));
}
*/

/*
mat3 pmatWood(vec2 st, vec3 colorA, vec3 colorB)
{
    // from https://thebookofshaders.com/edit.php#11/wood.frag
    vec2 pos = st.yx * vec2(10.0, 3.0);

    float pattern = pos.x;

    // Add noise
    pos = rotate2d(generateNoise(pos)) * pos;

    // Draw lines
    pattern = generateLines(pos, 0.5);

    return mat3(colorA * pattern, vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));
}
*/

vec3 gammaCorrection(vec3 color)
{
    float exp = 1.0 / 2.2;
    return vec3(
        pow(color.r, exp),
        pow(color.g, exp),
        pow(color.b, exp)
    );
}

vec3 gammaCorrectionInverse(vec3 color)
{
    float exp = 2.2;
    return vec3(
        pow(color.r, exp),
        pow(color.g, exp),
        pow(color.b, exp)
    );
}

vec3 flattenColor(vec3 color, int colors)
{
    float divider = float(colors);
    return round(color * divider) / divider;
}

vec3 getLightLocation(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos, 0), 0).xyz;
}

vec3 getLightColor(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos + 1, 0), 0).rgb;
}

float getLightRange(int n)
{
    int pos = n * 3;
    return texelFetch(lightsData, ivec2(pos + 2, 0), 0).r;
}

mat4 getObjectTrafo(WorldLocator loc)
{
    mat4 cm = cubeTrafo(loc.cube);
    vec3 p = resolveObjectLocator(loc.object);
    mat4 om = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(p, 1.0)
    );

    return cm * om;
}

mat4 getObjectInverseTrafo(WorldLocator loc)
{
    mat4 cm = cubeTrafoInverse(loc.cube);
    vec3 p = resolveObjectLocator(loc.object);
    mat4 om = mat4(
        vec4(1.0, 0.0, 0.0, 0.0),
        vec4(0.0, 1.0, 0.0, 0.0),
        vec4(0.0, 0.0, 1.0, 0.0),
        vec4(-p, 1.0)
    );

    return cm * om;
}

vec3 refr(vec3 ray, vec3 surfaceNormal, float ior)
{
    float eta = 1.0 / ior;
    float cosi = clamp(dot(surfaceNormal, ray), -1.0, 1.0);

    if (cosi > 0.0)
    {
        // exiting material, flipping around
        surfaceNormal *= -1.0;
        ior = 1.0 / ior;
        cosi = -cosi;
    }
    else
    {
        // entering material
    }
    float k = 1.0 - eta * eta * (1.0 - (cosi * cosi));
    if (k < 0.0)
    {
        // no refraction possible (total internal reflection)
        return vec3(0.0);
    }
    else
    {
        vec3 t1 = ray * eta;
        float t2 = cosi * eta + sqrt(k);
        return t1 - surfaceNormal * t2;
    }
}

/* Transforms a world-space point into object space.
 */
vec3 transformPoint(vec3 p, WorldLocator obj)
{
    mat4 m = getObjectInverseTrafo(obj);
    return (m * vec4(p, 1.0)).xyz;    
}

/* Transforms a surface normal in object space into world space.
 */
vec3 transformNormalOW(vec3 normal, WorldLocator obj)
{
    mat4 trafo = getObjectTrafo(obj);
    vec3 objLocW = (trafo * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 surfaceLocW = (trafo * vec4(normal, 1.0)).xyz;
    return normalize(surfaceLocW - objLocW);
}

/*
float dot2(vec2 v)
{
    return dot(v,v);
}

float dot2(vec3 v)
{
    return dot(v,v);
}

float ndot(vec2 a, vec2 b)
{
    return a.x * b.x - a.y * b.y;
}
*/

float sdfBox(vec3 p)
{
    vec3 halfSides = vec3(0.5);
    vec3 pt = p - vec3(0.0);
    vec3 q = abs(pt) - halfSides;
    return length(max(q, 0.0)) - min(0.0, max(max(q.x, q.y), q.z));
}

/* Processes a set of TASM instructions to generate a texture.
 */
mat3 processTasm(int program, vec2 st, vec3 p, float travelDist)
{
    // Since the GPU is quite limited on what it can do, implementing the
    // TASM instruction set might be too heavy for it. Therefore, all TASM
    // instructions are broken down into microcode defined by the TASM firmware.
    // The GPU processes the microcode only.

    bool programTooLong = false;
    bool stackOutOfBounds = false;

    int batchSize = 0;
    int srcPointer = 0;
    int destPointer = 0;
    int offsets = 0;
    int srcOffset = 0;
    int destOffset = 0;

    float workParam1 = 0.0;
    float workParam2 = 0.0;
    float workParam3 = 0.0;
    float workParam4 = 0.0;
    
    float v = 0.0;
    vec3 resultVec;

    float instructionSize = 0.0;
    int op = 0;
    int opCode = 0;
    int ri;

    vec4 instruction;
    vec4 microCodeCopyReg1;
    vec4 microCodeTest;
    vec4 microCodeBinOp;
    vec4 microCodeGenOp;
    vec4 microCodeAddReg;
    vec4 microCodeCopyReg2;

    // initialize
    tasmRegisters[REG_PC] = 0.0;
    tasmRegisters[REG_SP] = float(REG_STACK);

    tasmRegisters[REG_COLOR_R] = 1.0;
    tasmRegisters[REG_COLOR_G] = 0.0;
    tasmRegisters[REG_COLOR_B] = 0.0;

    tasmRegisters[REG_NORMAL_X] = 0.0;
    tasmRegisters[REG_NORMAL_Y] = 0.0;
    tasmRegisters[REG_NORMAL_Z] = 1.0;

    tasmRegisters[REG_ATTRIB_1] = 1.0;
    tasmRegisters[REG_ATTRIB_2] = 0.0;
    tasmRegisters[REG_ATTRIB_3] = 0.0;

    tasmRegisters[REG_ENV_TIMEMS] = float(timems);
    tasmRegisters[REG_ENV_ST_X] = st.x;
    tasmRegisters[REG_ENV_ST_Y] = st.y;
    tasmRegisters[REG_ENV_RAY_DISTANCE] = travelDist;
    tasmRegisters[REG_ENV_P_X] = p.x;
    tasmRegisters[REG_ENV_P_Y] = p.y;
    tasmRegisters[REG_ENV_P_Z] = p.z;

    for (int i = 0; i < 96; ++i)
    {
        programTooLong = i == 95;
        stackOutOfBounds = tasmRegisters[REG_SP] < float(REG_STACK) ||
                           tasmRegisters[REG_SP] >= float(REG_USER);

        if (tasmRegisters[REG_PC] < 0.0 || programTooLong || stackOutOfBounds)
        {
            // exit
            break;
        }

        instruction = texelFetch(tasmData, ivec2(int(tasmRegisters[REG_PC]), program), 0);

        opCode = int(instruction.r);
        instructionSize = instruction.g;
        tasmRegisters[REG_PARAM1] = instruction.b;
        tasmRegisters[REG_PARAM2] = instruction.a;

        // caching these appears to add too much overhead and memory spilling, and we're generally
        // better off without caching
        microCodeCopyReg1 = texelFetch(tasmData, ivec2(0, 3000 + opCode), 0);
        microCodeTest = texelFetch(tasmData, ivec2(1, 3000 + opCode), 0);
        microCodeBinOp = texelFetch(tasmData, ivec2(2, 3000 + opCode), 0);
        microCodeGenOp = texelFetch(tasmData, ivec2(3, 3000 + opCode), 0);
        microCodeAddReg = texelFetch(tasmData, ivec2(4, 3000 + opCode), 0);
        microCodeCopyReg2 = texelFetch(tasmData, ivec2(5, 3000 + opCode), 0);

        // advance program counter
        tasmRegisters[REG_PC] += instructionSize;

        // copy n registers from *source to *dest (avoid with batchSize = 0)
        batchSize = int(microCodeCopyReg1.r);
        srcPointer = int(tasmRegisters[int(microCodeCopyReg1.g)]);
        destPointer = int(tasmRegisters[int(microCodeCopyReg1.b)]);
        offsets = int(microCodeCopyReg1.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;
        for (ri = 0; ri < 3; ++ri)
        {
            tasmRegisters[destPointer + ri + destOffset] = ri < batchSize ? tasmRegisters[srcPointer + ri + srcOffset]
                                                                          : tasmRegisters[destPointer + ri + destOffset];
        }

        // test
        op = int(microCodeTest.r);
        if (op > 0)
        {
            workParam1 = tasmRegisters[int(tasmRegisters[REG_SP]) - 2];
            workParam2 = tasmRegisters[int(tasmRegisters[REG_SP]) - 1];
            tasmRegisters[REG_PC] = (op == 1 && workParam1 < workParam2) ||
                                (op == 2 && workParam1 <= workParam2) ||
                                (op == 3 && abs(workParam1 - workParam2) < 0.0001) ||
                                (op == 4 && workParam1 > workParam2) ||
                                (op == 5 && workParam1 >= workParam2)
                              ? tasmRegisters[REG_PC]
                              : tasmRegisters[REG_PARAM1];
        }

        // binop
        op = intr(microCodeBinOp.r);
        if (op > 0)
        {
            batchSize = intr(microCodeBinOp.g);
            for (ri = 0; ri < 3; ++ri)
            {
                workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2 * batchSize + ri];
                workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - batchSize + ri];
                v = (op == 1) ? workParam1 + workParam2 : v;
                v = (op == 2) ? workParam1 - workParam2 : v;
                v = (op == 3) ? workParam1 * workParam2 : v;
                v = (op == 4) ? workParam1 / workParam2 : v;
                v = (op == 5) ? min(workParam1, workParam2) : v;
                v = (op == 6) ? max(workParam1, workParam2) : v;
                v = (op == 7) ? workParam1 + exp(workParam2) : v;
                tasmRegisters[intr(tasmRegisters[REG_SP]) - 2 * batchSize + ri] = ri < batchSize ? v
                                                                                                 : workParam1;
            }
        }

        // gen
        op = intr(microCodeGenOp.r);
        if (op == 5)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];
            workParam4 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 4];

            resultVec = generateBumpNormal(workParam4, workParam3, workParam2, workParam1);
            
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 4] = resultVec.x;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 3] = resultVec.y;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 2] = resultVec.z;
        }
        else if (op == 6)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];

            resultVec = vec3(generateMipMap(vec2(workParam3, workParam2), int(workParam1)), 0.0);
            
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 3] = resultVec.x;
            tasmRegisters[intr(tasmRegisters[REG_SP]) - 2] = resultVec.y;

        }
        else if (op > 0)
        {
            workParam1 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 1];
            workParam2 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 2];
            workParam3 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 3];
            workParam4 = tasmRegisters[intr(tasmRegisters[REG_SP]) - 4];

            tasmRegisters[REG_PARAM1] = (op == 1 ? generateLine(vec2(workParam4, workParam3), workParam2, workParam1) : 0.0) +
                                        (op == 2 ? generateCheckerboard(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 3 ? generateWhiteNoise(vec2(workParam2, workParam1)) : 0.0) + 
                                        (op == 4 ? generateCellularNoise2D(vec2(workParam4, workParam3), int(workParam2), workParam1) : 0.0);
        }

        // add const value to a register (avoid with void pointer)
        srcPointer = int(tasmRegisters[int(microCodeAddReg.r)]);
        tasmRegisters[srcPointer] += microCodeAddReg.g;

        // copy n registers from *source to *dest (avoid with batch size = 0)
        batchSize = intr(microCodeCopyReg2.r);
        srcPointer = intr(tasmRegisters[int(microCodeCopyReg2.g)]);
        destPointer = intr(tasmRegisters[int(microCodeCopyReg2.b)]);
        offsets = int(microCodeCopyReg2.a);
        srcOffset = (offsets >> 4) - 8;
        destOffset = (offsets & 15) - 8;
        for (ri = 0; ri < 3; ++ri)
        {
            tasmRegisters[destPointer + ri + destOffset] = ri < batchSize ? tasmRegisters[srcPointer + ri + srcOffset]
                                                                          : tasmRegisters[destPointer + ri + destOffset];
        }
    }

    return mat3(
        vec3(
            programTooLong || stackOutOfBounds ? 1.0 : tasmRegisters[REG_COLOR_R],
            programTooLong ? 0.0 : stackOutOfBounds ? 1.0 : tasmRegisters[REG_COLOR_G],
            programTooLong ? 1.0 : stackOutOfBounds ? 0.0 : tasmRegisters[REG_COLOR_B]
        ),
        vec3(tasmRegisters[REG_NORMAL_X], tasmRegisters[REG_NORMAL_Y], tasmRegisters[REG_NORMAL_Z]),
        vec3(tasmRegisters[REG_ATTRIB_1], tasmRegisters[REG_ATTRIB_2], tasmRegisters[REG_ATTRIB_3])
    );
}

bool isEdgeZ(vec3 p)
{
    // p is in object space
    p = abs(p);
    return p.x > 0.4999 && abs(p.x - p.y) < 0.0001;
}

bool isEdgeY(vec3 p)
{
    // p is in object space
    p = abs(p);
    return p.x > 0.4999 && abs(p.x - p.z) < 0.0001;
}

bool isEdgeX(vec3 p)
{
    // p is in object space
    p = abs(p);
    return p.y > 0.4999 && abs(p.y - p.z) < 0.0001;
}

vec3 getSurfaceNormal(vec3 p)
{
    // p is in object space

    vec3 d = abs(p);
    if (d.x > d.y && d.x > d.z)
    {
        return vec3(sign(p.x), 0.0, 0.0);
    }
    if (d.y > d.z)
    {
        return vec3(0.0, sign(p.y), 0.0);
    }
    return vec3(0.0, 0.0, sign(p.z));


    // this gives rounded edges...
    /*
    // move p a step away from the surface to not fall into the object
    p *= 1.001;
    float epsilon = 0.05;
    return normalize(
        vec3(
            sdfBox(p + vec3(epsilon, 0.0, 0.0)) - sdfBox(p + vec3(-epsilon, 0.0, 0.0)),
            sdfBox(p + vec3(0.0, epsilon, 0.0)) - sdfBox(p + vec3(0.0, -epsilon, 0.0)),
            sdfBox(p + vec3(0.0, 0.0, epsilon)) - sdfBox(p + vec3(0.0, 0.0, -epsilon))
        )
    );
    */
}

/* Returns the surface material at the given location as a mat3:
 *
 * - vec3: color
 * - vec3: normal vector (z pointing upwards)
 * - vec3: roughness, ior, volumetric
 */
mat3 getObjectMaterial(WorldLocator obj, vec3 p, vec3 worldP, float travelDist)
{
    int materialId = objectDataEntry(obj);
    vec2 st = p.xy;

    // position texture on cube
    vec3 n = getSurfaceNormal(p);
    vec3 p2 = abs(n.y) > 0.0 ? n.zxy : n.zyx;
    float dp = dot(n, p2);
    vec3 axis1 = normalize(p2 - dp * n);
    vec3 axis2 = normalize(cross(n, axis1));
    float x = dot(p, axis1);
    float y = dot(p, axis2);
    st = 0.5 + vec2(x, y);

    return enableTasm ? processTasm(materialId, st, worldP, travelDist)
                      : mat3(
                            vec3(1.0),
                            vec3(0.0, 0.0, 1.0),
                            vec3(1.0, 0.0, 0.0)
                        );
}

bool cubeHasObject(ObjectLocator objLoc, ivec4 pattern, int lod)
{
    int cubeLod = getCubeLod(lod);
    int bitsPerCoord = 2 / (1 << cubeLod);

    int patternHi = pattern.r;
    int patternHiMid = pattern.g;
    int patternLoMid = pattern.b;
    int patternLo = pattern.a;

    ivec3 loc = ivec3(objLoc.x, objLoc.y, objLoc.z);
    loc /= (1 << cubeLod);

    int n = (loc.x << (bitsPerCoord + bitsPerCoord)) +
            (loc.y << bitsPerCoord) +
            loc.z;
    return n < 16 ? (patternLo & (1 << n)) > 0
                  : n < 32 ? (patternLoMid & (1 << (n - 16))) > 0
                           : n < 48 ? (patternHiMid & (1 << (n - 32))) > 0
                                    : (patternHi & (1 << (n - 48))) > 0;
}

bool hasObjectAt(vec3 p)
{
    CubeLocator cube = makeSuperCubeLocator(p, 0);
    mat4 m = cubeTrafoInverse(cube);
    vec3 pT = (m * vec4(p, 1.0)).xyz;
    ObjectLocator objLoc = makeObjectLocator(pT);

    ivec2 offset = cubeDataOffset(cube);
    ivec4 pattern = texelFetch(worldData, offset, 0);

    return cubeHasObject(objLoc, pattern, lodOfSector(cube.sector));
}

ObjectAndDistance raymarchVoxels(CubeLocator cube, vec3 origin, vec3 entryPoint, vec3 rayDirection)
{
    WorldLocator noObject;

    ivec2 offset = cubeDataOffset(cube);
    ivec4 pattern = texelFetch(worldData, offset, 0);
    if (pattern.r + pattern.g + pattern.b + pattern.a == 0)
    {
        // this cube is empty
        return ObjectAndDistance(noObject, 9999.0);
    }

    mat4 m = cubeTrafoInverse(cube);
    vec3 p = entryPoint;
    vec3 pT = (m * vec4(p, 1.0)).xyz;

    if (pT.x < 0.0 || pT.y < 0.0 || pT.z < 0.0 ||
        pT.x >= 4.0 || pT.y >= 4.0 || pT.z >= 4.0)
    {
        // entry point is out of bounds
        debug = 1;
        debugColor = vec3(1.0, 1.0, 0.0);
        return ObjectAndDistance(noObject, 9999.0);
    }

    float gridSize = 1.0;

    ObjectLocator objLoc = makeObjectLocator(pT);
    if (cubeHasObject(objLoc, pattern, lodOfSector(cube.sector)))
    {
        WorldLocator obj = makeWorldLocator(cube, objLoc);
        voxelDistanceMap = distance(origin, resolveCubeLocator(obj.cube) + resolveObjectLocator(obj.object));
        return ObjectAndDistance(obj, distance(origin, p));
    }

    vec3 scalingsOnGrid = vec3(
        rayDirection.x != 0.0 ? 1.0 / abs(rayDirection.x) : 9999.0,
        rayDirection.y != 0.0 ? 1.0 / abs(rayDirection.y) : 9999.0,
        rayDirection.z != 0.0 ? 1.0 / abs(rayDirection.z) : 9999.0
    );

    vec3 disabler = vec3(
        scalingsOnGrid.x < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.y < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.z < 9990.0 ? 0.0 : 9999.0
    );

    vec3 gridPoint = floor(p / gridSize) * gridSize;

    gridPoint += vec3(
        rayDirection.x > 0.0 ? 1.0 : 0.0,
        rayDirection.y > 0.0 ? 1.0 : 0.0,
        rayDirection.z > 0.0 ? 1.0 : 0.0
    ) * gridSize;

    vec3 distsOnGrid = abs(gridPoint - p);
    for (int i = 0; i < 12; ++i)
    {
        vec3 rayLengths = distsOnGrid * scalingsOnGrid + disabler;
        bool advanceX = rayLengths.x <= rayLengths.y && rayLengths.x <= rayLengths.z;
        bool advanceY = rayLengths.y <= rayLengths.x && rayLengths.y <= rayLengths.z;
        bool advanceZ = rayLengths.z <= rayLengths.x && rayLengths.z <= rayLengths.y;

        if (advanceX && advanceZ)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceX && advanceY)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceY && advanceZ)
        {
            advanceX = false;
            advanceZ = false;
        }

        vec3 advanceVec = vec3(
            advanceX ? sign(rayDirection.x) : 0.0,
            advanceY ? sign(rayDirection.y) : 0.0,
            advanceZ ? sign(rayDirection.z) : 0.0
        ) * gridSize;

        distsOnGrid += abs(advanceVec);
        vec3 epsilon = advanceVec * 0.00001;

        // be sure to take only one of the rayLengths
        p = entryPoint + rayDirection * ((advanceX ? rayLengths.x : 0.0) +
                                         (advanceY ? rayLengths.y : 0.0) +
                                         (advanceZ ? rayLengths.z : 0.0)) + epsilon;

        vec3 pT = (m * vec4(p, 1.0)).xyz;
        if (pT.x < 0.0 || pT.y < 0.0 || pT.z < 0.0 ||
            pT.x >= 4.0 || pT.y >= 4.0 || pT.z >= 4.0)
        {
            // leaving the cube
            break;
        }

        ObjectLocator objLoc = makeObjectLocator(pT);
        if (cubeHasObject(objLoc, pattern, lodOfSector(cube.sector)))
        {
            WorldLocator obj = makeWorldLocator(cube, objLoc);
            voxelDistanceMap = distance(origin, resolveCubeLocator(obj.cube) + resolveObjectLocator(obj.object));
            return ObjectAndDistance(obj, distance(origin, p));
        }
    }

    return ObjectAndDistance(noObject, 9999.0);
}

ObjectAndDistance raymarchCubes(vec3 origin, vec3 rayDirection, int depth, float maxDistance)
{
    WorldLocator noObject;
    ObjectAndDistance result = ObjectAndDistance(noObject, 9999.0);

    float gridSize = 4.0;
    vec3 p = origin;

    CubeLocator originCube = makeSuperCubeLocator(p, 0);
    result = raymarchVoxels(originCube, origin, p, rayDirection);
    if (result.distance < 9999.0)
    {
        return result;
    }

    vec3 scalingsOnGrid = vec3(
        rayDirection.x != 0.0 ? 1.0 / abs(rayDirection.x) : 9999.0,
        rayDirection.y != 0.0 ? 1.0 / abs(rayDirection.y) : 9999.0,
        rayDirection.z != 0.0 ? 1.0 / abs(rayDirection.z) : 9999.0
    );

    vec3 disabler = vec3(
        scalingsOnGrid.x < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.y < 9990.0 ? 0.0 : 9999.0,
        scalingsOnGrid.z < 9990.0 ? 0.0 : 9999.0
    );

    vec3 gridPoint = floor(p / gridSize) * gridSize;

    gridPoint += vec3(
        rayDirection.x > 0.0 ? 1.0 : 0.0,
        rayDirection.y > 0.0 ? 1.0 : 0.0,
        rayDirection.z > 0.0 ? 1.0 : 0.0
    ) * gridSize;

    vec3 distsOnGrid = abs(gridPoint - p);
    int i = 0;
    for (; i < depth; ++i)
    {
        vec3 rayLengths = distsOnGrid * scalingsOnGrid + disabler;
        bool advanceX = rayLengths.x <= rayLengths.y && rayLengths.x <= rayLengths.z;
        bool advanceY = rayLengths.y <= rayLengths.x && rayLengths.y <= rayLengths.z;
        bool advanceZ = rayLengths.z <= rayLengths.x && rayLengths.z <= rayLengths.y;

        if (advanceX && advanceZ)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceX && advanceY)
        {
            advanceY = false;
            advanceZ = false;
        }
        if (advanceY && advanceZ)
        {
            advanceX = false;
            advanceZ = false;
        }

        vec3 advanceVec = vec3(
            advanceX ? sign(rayDirection.x) : 0.0,
            advanceY ? sign(rayDirection.y) : 0.0,
            advanceZ ? sign(rayDirection.z) : 0.0
        ) * gridSize;

        if (! advanceX && ! advanceZ && advanceY && (length(advanceVec) == 0.0))
        {
            debug = 1;
            debugColor = vec3(rayLengths.z < 100.0 ? 1.0 : 0.0, 0.0, 0.0);
        }
        else
        {
            debug = 0;
        }
        //if (length(advanceVec) == 0.0) debug = 2;

        distsOnGrid += abs(advanceVec);
        if (abs(advanceVec.y) > 0.0) debug = 0;
        vec3 epsilon = advanceVec * 0.00001;

        // be sure to take only one of the rayLengths
        p = origin + rayDirection * ((advanceX ? rayLengths.x : 0.0) +
                                     (advanceY ? rayLengths.y : 0.0) +
                                     (advanceZ ? rayLengths.z : 0.0)) + epsilon;

        cubeEntryDistanceMap = max(cubeEntryDistanceMap, distance(origin, p));

        if (p.x < 0.0 || p.y < 0.0 || p.z < 0.0 ||
            p.x >= float(sectorSize * cubeSize * horizonSize) || p.y >= float(sectorSize * cubeSize * horizonSize) || p.z >= float(sectorSize * cubeSize * horizonSize) ||
            distance(p, origin) > maxDistance)
        {
            // out of view
            break;
        }

        CubeLocator cube = makeSuperCubeLocator(p, 0);
        result = raymarchVoxels(cube, origin, p, rayDirection);
        if (result.distance < 9999.0)
        {
            break;
        }
    }

    cubeDistanceMap = cubeDistanceMap > 0 ? cubeDistanceMap : i;
    return result;
}

ObjectAndDistance raymarch(vec3 origin, vec3 rayDirection, float maxDistance)
{
    return raymarchCubes(origin, rayDirection, marchingDepth, maxDistance);
}

vec4 skyBox(vec3 origin, vec3 rayDirection)
{
    vec3 hitPoint = abs(rayDirection.y) > 0.001 ? origin + rayDirection * ((1000.0 - origin.y) / rayDirection.y)
                                                : origin + rayDirection;
    // TODO: move to TASM
    vec3 timedHitPoint = hitPoint + vec3(float(timems) / 30000.0, 0.0, 0.0);

    vec3 color = enableTasm ? processTasm(0, (timedHitPoint.xz / 10000.0), hitPoint, distance(origin, hitPoint))[0]
                            : vec3(0.0);

    return vec4(color, 1.0);
}

vec3 simplePhongShading(vec3 checkPoint)
{
    vec3 lighting = vec3(0.0);

    for (int i = 0; i < 3; ++i)
    {
        if (i == numLights)
        {
            break;
        }

        vec3 lightLoc = getLightLocation(i);
        vec3 lightCol = getLightColor(i);
        float lightRange = getLightRange(i);

        vec3 directionToLight = normalize(lightLoc - checkPoint);
        float lightDistance = length(checkPoint - lightLoc);

        // light attenuation based on distance and strength of the light source
        float attenuation = clamp(1.0 - lightDistance / lightRange, 0.0, 1.0);
        attenuation *= attenuation;
        vec3 attenuatedLight = lightCol * attenuation;

        // diffuse light
        float diffuseImpact = 1.0;
        vec3 diffuse = attenuatedLight * diffuseImpact;

        lighting += diffuse;
    }

    return lighting.rgb;
}

vec3 phongShading(vec3 origin, vec3 checkPoint, vec3 surfaceNormal, float roughness)
{
    // Phong shading: lighting = ambient + diffuse + specular
    //                color = modelColor * lighting

    vec3 viewDirection = normalize(origin - checkPoint);
    vec3 lighting = vec3(0.2);
    float shininess = (1.0 - roughness) * 64.0;

    for (int i = 0; i < 3; ++i)
    {
        if (i == numLights)
        {
            break;
        }

        vec3 lightLoc = getLightLocation(i);
        vec3 lightCol = getLightColor(i);
        float lightRange = getLightRange(i);

        vec3 directionToLight = normalize(lightLoc - checkPoint);
        float lightDistance = length(checkPoint - lightLoc);

        float impact = dot(directionToLight, surfaceNormal);

        // does the light reach?
        if (impact <= 0.001)
        {
            // nope
            continue;
        }
        if (enableShadows)
        {
            // we may not have to go all the way to the light to know if it reaches (it may be far far away)
            float optimizedLightDistance = min(lightDistance, 100.0);
            float travelDist = raymarch(checkPoint + directionToLight * 0.1, directionToLight, optimizedLightDistance).distance;
            if (travelDist < optimizedLightDistance)
            {
                // nope
                continue;
            }
        }

        // light attenuation based on distance and strength of the light source
        float attenuation = clamp(1.0 - lightDistance / lightRange, 0.0, 1.0);
        if (i == 0)
        {
            // light attenuation based on clouds
            vec4 skyColor = skyBox(checkPoint, directionToLight);
            attenuation *= lerp(0.1, 1.0, 1.0 - skyColor.r);
        }

        attenuation *= attenuation;
        vec3 attenuatedLight = lightCol * attenuation;

        // diffuse light
        vec3 diffuse = attenuatedLight * impact;

        // specular highlight (Blinn-Phong)
        vec3 halfDirection = normalize(directionToLight + viewDirection);
        float specularStrength = pow(max(0.0, dot(surfaceNormal, halfDirection)), shininess) * 0.5;
        vec3 specular = attenuatedLight * specularStrength;

        lighting += diffuse + specular;
    }

    return lighting.rgb;
}

float ambientOcclusion(vec3 checkPoint, vec3 rayDirection, WorldLocator obj, mat4 surfaceTrafo, float size)
{
    vec3 checkPoint2 = checkPoint - rayDirection * 0.01;
    vec3 checkPointT2 = transformPoint(checkPoint2, obj);
    vec3 surfacePoint = clamp((inverse(surfaceTrafo) * vec4(checkPointT2, 1.0)).xyz, -0.5, 0.5);
    mat4 aoTrafo = getObjectTrafo(obj);
    vec3 v1 = (surfaceTrafo * vec4(size, 0.0, 0.0, 1.0)).xyz;
    vec3 v2 = (surfaceTrafo * vec4(0.0, size, 0.0, 1.0)).xyz;

    vec3 p1 = (aoTrafo * vec4(checkPointT2 + v1, 1.0)).xyz;
    vec3 p2 = (aoTrafo * vec4(checkPointT2 - v1, 1.0)).xyz;
    vec3 p3 = (aoTrafo * vec4(checkPointT2 + v2, 1.0)).xyz;
    vec3 p4 = (aoTrafo * vec4(checkPointT2 - v2, 1.0)).xyz;

    bool hasP1 = hasObjectAt(p1);
    bool hasP2 = hasObjectAt(p2);
    bool hasP3 = hasObjectAt(p3);
    bool hasP4 = hasObjectAt(p4);

    float dist1 = 0.5 - surfacePoint.x;
    float dist2 = surfacePoint.x + 0.5;
    float dist3 = 0.5 - surfacePoint.y;
    float dist4 = surfacePoint.y + 0.5;
    
    float shadow = ((hasP1 ? size - dist1 : 0.0) +
                    (hasP2 ? size - dist2 : 0.0) +
                    (hasP3 ? size - dist3 : 0.0) +
                    (hasP4 ? size - dist4 : 0.0)) / (2.0 * size);

    bool hasAo = shadow > 0.0;
    aoEdge = hasAo;

    return clamp(shadow, 0.0, 1.0);
}

/* Determining the box normals is tricky around the edges and corners.
 * To get this right, we have to check their surroundings.
 */
vec3 getCorrectedBoxNormals(WorldLocator obj, vec3 p, vec3 rayDirection)
{
    // p is in object space

    // we have to check the neighbors to resolve surface normal ambiguities at the edges
    vec3 centerPoint = (getObjectTrafo(obj) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    // the three normals facing towards the camera
    vec3 surfaceNormalX = vec3(sign(p.x), 0.0, 0.0);
    vec3 surfaceNormalY = vec3(0.0, sign(p.y), 0.0);
    vec3 surfaceNormalZ = vec3(0.0, 0.0, sign(p.z));

    // we have to check the dot products for negativity to filter out normals facing away
    float dotX = dot(surfaceNormalX, -rayDirection);
    float dotY = dot(surfaceNormalY, -rayDirection);
    // dotZ not required

    // check if there are neighbors on sides facing to the camera,
    // as these sides cannot be visible
    bool hasXNeighbor = hasObjectAt(centerPoint + surfaceNormalX);
    bool hasYNeighbor = hasObjectAt(centerPoint + surfaceNormalY);
    bool hasZNeighbor = hasObjectAt(centerPoint + surfaceNormalZ);

    // the interesting (because ambiguous) places are the edges and corners
    bool edgeX = isEdgeX(p);
    bool edgeY = isEdgeY(p);
    bool edgeZ = isEdgeZ(p);
    
    // side-computation: only free edges may show toon lines
    freeEdge = edgeX && (! hasYNeighbor && ! hasZNeighbor) ||
               edgeY && (! hasXNeighbor && ! hasZNeighbor) ||
               edgeZ && (! hasXNeighbor && ! hasYNeighbor);

    vec3 surfaceNormal;

    // the easy case: two normals blocked, only one left
    if (hasXNeighbor && hasZNeighbor)
    {
        surfaceNormal = surfaceNormalY;
    }
    else if (hasXNeighbor && hasYNeighbor)
    {
        surfaceNormal = surfaceNormalZ;
    }
    else if (hasYNeighbor && hasZNeighbor)
    {
        surfaceNormal = surfaceNormalX;
    }

    // where two edges meet, there is a corner
    else if (edgeX && edgeY || edgeY && edgeZ || edgeX && edgeZ)
    {
        if (hasXNeighbor)
        {
            surfaceNormal = dotY > 0.0 && ! hasYNeighbor ? surfaceNormalY : surfaceNormalZ;
        }
        else if (hasYNeighbor)
        {
            surfaceNormal = dotX > 0.0 && ! hasXNeighbor ? surfaceNormalX : surfaceNormalZ;
        }
        else if (hasZNeighbor)
        {
            surfaceNormal = dotX > 0.0 && ! hasXNeighbor ? surfaceNormalX : surfaceNormalY;
        }
        else
        {
            surfaceNormal = dotX > 0.0 ? surfaceNormalX : dotY > 0.0 ? surfaceNormalY : surfaceNormalZ;
        }
    }
    else if (edgeZ)
    {
        //debug = 2;
        surfaceNormal = hasXNeighbor || dotX <= 0.0 ? surfaceNormalY : surfaceNormalX;
    }
    else if (edgeX)
    {
        surfaceNormal = hasYNeighbor || dotY <= 0.0 ? surfaceNormalZ : surfaceNormalY;
    }
    else if (edgeY)
    {
        surfaceNormal = hasXNeighbor || dotX <= 0.0 ? surfaceNormalZ : surfaceNormalX;
    }
    else
    {
        //debug = 2;
        surfaceNormal = getSurfaceNormal(p);
    }

    return surfaceNormal;
}

/* Returns the color at the given screen pixel plus the ID of the object that was
 * hit in the a component. The object ID is added to the amount of traces multiplied
 * by 1000.
 */
vec4 shootRayThroughScreen(vec2 uv, vec3 origin, float aspect)
{
    // transform the camera location and orientation
    vec3 currentOrigin = (cameraTrafo * vec4(origin, 1.0)).xyz;
    vec3 screenPoint = (cameraTrafo * vec4(uv.x, uv.y / aspect, 1.0, 1.0)).xyz;

    WorldLocator currentObject;

    // shoot a ray from origin onto the near Plain (screen)
    vec3 rayDirection = normalize(screenPoint - currentOrigin);

    float travelDistance = 0.0;

    vec3 color = vec3(1.0);
    vec3 light = vec3(0.0);
    vec3 volumetricColor = vec3(0.8);
    vec3 volumetricLight = vec3(0.0);
    float volumetricDensity = 0.0;

    bool insideObject = false;

    int traceCount = 0;
    for (; traceCount < 8; ++traceCount)
    {
        if (traceCount == tracingDepth)
        {
            // tracing depth exceeded
            //debug = 2;
            break;
        }

        ObjectAndDistance objectAndDist = raymarch(currentOrigin, rayDirection, 9999.0); // traceCount == 0 ? 9999.0 : 100.0);
        WorldLocator obj = objectAndDist.object;
        float dist = objectAndDist.distance;
        
        travelDistance += dist;
        currentObject = obj;

        depthMap = depthMap > 0.0 ? depthMap : dist;

        if (dist < 9999.0)
        {
            // hit something
            vec3 checkPoint = currentOrigin + rayDirection * dist;
            vec3 checkPointT = transformPoint(checkPoint, obj);

            mat3 materialData = getObjectMaterial(obj, checkPointT, checkPoint, travelDistance);
            //mat3 materialData = mat3(vec3(1.0), vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0));
            vec3 materialColor = materialData[0];
            float roughness = materialData[2].x;
            float reflectivity = 1.0 - roughness;
            float ior = materialData[2].y;
            float volumetric = materialData[2].z;

            vec3 surfaceNormal = getCorrectedBoxNormals(obj, checkPointT, rayDirection);

            //vec3 surfaceNormal = transformNormalOW(surfaceNormalT, obj);
            mat4 surfaceTrafo = createSurfaceTrafo(surfaceNormal);
            vec3 bumpNormalM = materialData[1];
            vec3 bumpNormalT = (surfaceTrafo * vec4(bumpNormalM, 1.0)).xyz;
            vec3 bumpNormal = transformNormalOW(bumpNormalT, obj);
           
            // for debugging: show normals
            materialColor = showNormals ? vec3(0.5) + bumpNormal * 0.5
                                        : materialColor;

            if (volumetric < 0.1 && ior < 0.001)
            {
                vec3 lightIntensity = phongShading(currentOrigin, checkPoint, bumpNormal, roughness);
                light += lightIntensity;
                color *= materialColor;

                lightIntensity *= enableAmbientOcclusion ? 1.0 - ambientOcclusion(checkPoint, rayDirection, obj, surfaceTrafo, 0.1) * 0.5
                                                         : 1.0;

                color *= lightIntensity;
            }
            
            // measure volumetrics
            for (int i = 0; i < 100; ++i)
            {
                if (float(i) * 0.1 > dist)
                {
                    break;
                }
                vec3 samplePoint = currentOrigin + rayDirection * float(i) * 0.1;
                float f = float((timems / 100000) % 500);
                float v = max(0.0, 8.0 - samplePoint.y) * fogDensity * (1.0 - generateCellularNoise3D(samplePoint / 16.0 + f * vec3(0.0, -0.005, 0.0), 20));
                volumetricDensity += v;
                if (v > 0.0001)
                {
                    //volumetricColor *= 0.9 * simplePhongShading(samplePoint);
                    //volumetricLight += exp(-v) * simplePhongShading(samplePoint) * vec3(0.01);
                }
            }

            if (ior > 0.01)
            {
                // we're not finished yet - refract the ray and enter or exit the object
                vec3 refractedRay = refr(rayDirection, bumpNormal, ior);
                if (length(refractedRay) < 0.0001)
                {
                    // total internal reflection
                    rayDirection = normalize(reflect(rayDirection, bumpNormal));
                }
                else
                {
                    insideObject = ! insideObject;
                    rayDirection = normalize(refractedRay);
                }
                currentOrigin = checkPoint + rayDirection * 0.01;
                //light *= 0.5;
            }
            else if (reflectivity > 0.1)
            {
                // we're not finished yet - reflect the ray
                float fresnel = pow(clamp(1.0 - dot(bumpNormal, rayDirection * -1.0), 0.5, 1.0), 1.0);
                //checkPoint = currentOrigin + rayDirection * (dist - 0.5);
                rayDirection = reflect(rayDirection, bumpNormal);
                // epsilon must be small for corners, or you'll get reflected far into another block
                //currentOrigin = checkPoint; //bumpNormal * 0.1 + /* bump off the surface a bit */
                currentOrigin = checkPoint + rayDirection * 0.01;
                if (hasObjectAt(currentOrigin))
                {
                    // stuck in an object, not good...
                    break;
                }
                light *= fresnel * reflectivity;
            }
            else if (volumetric > 0.1)
            {
                // convert origin and ray into object space
                vec3 rayP = currentOrigin + rayDirection;
                vec3 rayPT = transformPoint(rayP, obj);

                vec3 originT = transformPoint(currentOrigin, obj);
                vec3 rayDirectionT = rayPT - originT;

                vec3 checkPointT = transformPoint(checkPoint, obj);
                light *= distance(checkPointT, vec3(0.0)) / 1.0;

                vec3 entryExit = hitAabb(originT, rayDirectionT);
                vec3 entryPoint = currentOrigin + rayDirection * entryExit.s;
                vec3 entryPointT = originT + rayDirectionT * entryExit.s; // + vec3(0.1, 0.1, 0.1);
                vec3 exitPointT = originT + rayDirectionT * entryExit.t;
                float len = entryExit.t - entryExit.s;

                float sdfDist = 0.0;
                for (int i = 0; i < 50; ++i)
                {
                    vec3 samplePointT = entryPointT + rayDirectionT * sdfDist;
                    float s1 = length(samplePointT - vec3(-0.5 / 2.0, 0.0, 0.0)) - 0.5;
                    float s2 = length(samplePointT - vec3(+0.5 / 2.0, 0.0, 0.0)) - 0.5;
                    float d = max(s1, s2);
                    if (d < 0.001)
                    {
                        vec3 li = phongShading(currentOrigin, entryPoint + rayDirection * d, surfaceNormal, roughness);
                        light += li;
                        color *= vec3(1.0, 0.0, 0.0);
                        color *= li;
                        travelDistance += d;
                        break;
                    }
                    else if (d > 2.0)
                    {
                        currentOrigin = checkPoint + (len + 0.01) * rayDirection;
                        travelDistance += len + 0.01;
                    }
                    else
                    {
                        sdfDist += d;
                    }
                }

                /*
                volumetricColor *= vec3(0.3, 0.9, 0.9);
                for (float step = 0.0; step < 1.0; step += 0.01)
                {
                    vec3 samplePointT = entryPointT + step * length * rayDirectionT;
                    if (distance(samplePointT, vec3(0.0)) < 0.4)
                    {
                        volumetricDensity += 1.0 - procCellularNoise3D(0.5 + samplePointT / 2.0, 15);
                        break;
                    }
                }
                currentOrigin = checkPoint + (length + 0.01) * rayDirection;
                travelDistance += length + 0.01;
                */
            }
            else
            {
                break;
            }
        }
        else
        {
            // hit the sky box
            vec4 skyColor = skyBox(currentOrigin, rayDirection);
            color *= skyColor.rgb;
            light += vec3(0.5);

            // measure volumetrics
            volumetricDensity = 0.0;
            for (int i = 0; i < 100; ++i)
            {
                vec3 samplePoint = currentOrigin + rayDirection * float(i) * 0.1;
                float f = float((timems / 100000) % 500);
                float v = max(0.0, 8.0 - samplePoint.y) * fogDensity * (1.0 - generateCellularNoise3D(samplePoint / 16.0 + f * vec3(0.0, -0.005, 0.0), 20));
                volumetricDensity += v;
            }
            break;
        }

    }

    // finally apply the light and fog
    //color *= light;

    // lerp between volumetric color and color according to the density factor
    if (volumetricDensity > 0.00001)
    {
        volumetricDensity = clamp(volumetricDensity, 0.0, 1.0);
        volumetricColor = clamp(volumetricColor, 0.0, 1.0);
        color = vec3(
            lerp(color.r, volumetricColor.r, volumetricDensity),
            lerp(color.g, volumetricColor.g, volumetricDensity),
            lerp(color.b, volumetricColor.b, volumetricDensity)
        );
    }

    // fog is an ubiquituous volumetric body
    float fogFactor = exp(-travelDistance * fogDensity);
    vec3 fogColor = vec3(1.0); // + 0.1 * random2(uv));

    color = vec3(
        lerp(fogColor.r, color.r, fogFactor),
        lerp(fogColor.g, color.g, fogFactor),
        lerp(fogColor.b, color.b, fogFactor)
    );
    /*
    if (fogDensity > 0.00001)
    {
        //volumetricDensity += 1.0 - fogFactor;
        //volumetricColor = volumetricColor * fogColor * (1.0 - fogFactor);
    }
    */
    
    return vec4(color, float(traceCount)); // * 1000 + currentObject));    
}

void main()
{
    // initialize const TASM registers
    tasmRegisters[REG_PTR_END_VALUE] = float(REG_END_VALUE);
    tasmRegisters[REG_PTR_VOID] = float(REG_VOID);
    tasmRegisters[REG_PTR_PC] = float(REG_PC);
    tasmRegisters[REG_PTR_SP] = float(REG_SP);
    tasmRegisters[REG_PTR_PARAM1] = float(REG_PARAM1);
    tasmRegisters[REG_PTR_PARAM2] = float(REG_PARAM2);
    tasmRegisters[REG_PTR_COLOR] = float(REG_COLOR_R);
    tasmRegisters[REG_PTR_NORMAL] = float(REG_NORMAL_X);
    tasmRegisters[REG_PTR_ATTRIBUTES] = float(REG_ATTRIB_1);
    tasmRegisters[REG_PTR_ATTRIB_2] = float(REG_ATTRIB_2);
    tasmRegisters[REG_PTR_ATTRIB_3] = float(REG_ATTRIB_3);
    tasmRegisters[REG_END_VALUE] = -1.0;
    tasmRegisters[REG_ENV_UNIVERSE_X] = universeLocation.x;
    tasmRegisters[REG_ENV_UNIVERSE_Y] = universeLocation.y;
    tasmRegisters[REG_ENV_UNIVERSE_Z] = universeLocation.z;

    if (screenWidth < 16.0 || screenHeight < 16.0)
    {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    float aspect = screenWidth / screenHeight;
    vec2 pixelSize = vec2(1.0) / vec2(screenWidth, screenHeight);

    float exposure = 1.0;

    vec3 origin = vec3(0, 0, -0.1);

    vec4 samplePoint = shootRayThroughScreen(uv, origin, aspect);
    vec3 pixel = samplePoint.rgb;
    pixel *= exposure;
    pixel = gammaCorrection(pixel);

    if (enableToonEffect)
    {   
        pixel = freeEdge || aoEdge ? vec3(0.0, 0.0, 0.0) : flattenColor(pixel, 8);
    }

    if (debug == 1)
    {
        fragColor = vec4(debugColor, 1.0);
    }
    else if (debug == 2)
    {
        fragColor = vec4(1.0, 0.0, 1.0, 1.0);
    }
    else if (debug == 3)
    {
        fragColor = vec4(1.0, 1.0, 0.0, 1.0);
    }
    else
    {
        fragColor = vec4(pixel, 1.0);
        //fragColor = vec4(vec3(float(min(cubeDistanceMap, 100)) / 100.0), 1.0);
        //fragColor = vec4(vec3(min(cubeEntryDistanceMap, 100.0) / 100.0), 1.0);
        //fragColor = vec4(vec3(min(depthMap, 100.0) / 100.0), 1.0);
        //fragColor = vec4(vec3(min(voxelDistanceMap, 100.0) / 100.0), 1.0);
    }
}
