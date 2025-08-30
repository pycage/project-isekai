export function mandelbrot(x: f64, y: f64, maxIterations: i32): i32
{
    let nextP1: f64 = 0.0;
    let nextP2: f64 = 0.0;
    let prevP1: f64 = 0.0;
    let prevP2: f64 = 0.0;

    let i: i32 = 0;
    for (; i < maxIterations; ++i)
    {
        prevP1 = nextP1;
        prevP2 = nextP2;
        
        nextP1 = prevP1 * prevP1 - prevP2 * prevP2 + x;
        nextP2 = 2.0 * prevP1 * prevP2 + y;

        if (nextP1 * nextP2 + nextP2 * nextP2 > 4.0)
        {
            break;
        }
    }

    return i;
}
