// Exact clipped Chebyshev distances from binary seeds. An integer summed-area
// table answers whether a square contains a seed; binary search finds the
// smallest radius. Both fields are independent and use no floating-point math.
struct Params { width:u32, height:u32, cap:u32, padding:u32 }
@group(0) @binding(0) var<storage,read> seeds:array<u32>;
@group(0) @binding(1) var<storage,read_write> prefix:array<vec2u>;
@group(0) @binding(2) var<storage,read_write> result:array<u32>;
@group(0) @binding(3) var<uniform> params:Params;

@compute @workgroup_size(64)
fn rows(@builtin(global_invocation_id) id:vec3u) {
    let y=id.x;
    if(y>=params.height) {return;}
    var sum=vec2u(0u);
    for(var x=0u;x<params.width;x++) {
        let i=y*params.width+x;let value=seeds[i];
        sum+=vec2u(value&1u,(value>>1u)&1u);prefix[i]=sum;
    }
}
@compute @workgroup_size(64)
fn columns(@builtin(global_invocation_id) id:vec3u) {
    let x=id.x;
    if(x>=params.width) {return;}
    var sum=vec2u(0u);
    for(var y=0u;y<params.height;y++) {
        let i=y*params.width+x;sum+=prefix[i];prefix[i]=sum;
    }
}
fn at(x:i32,y:i32)->vec2u {
    if(x<0 || y<0) {return vec2u(0u);}
    return prefix[u32(y)*params.width+u32(x)];
}
fn count(p:vec2i,r:u32)->vec2u {
    let lo=max(p-vec2i(i32(r)),vec2i(0));
    let hi=min(p+vec2i(i32(r)),vec2i(i32(params.width)-1,i32(params.height)-1));
    return at(hi.x,hi.y)+at(lo.x-1,lo.y-1)-at(lo.x-1,hi.y)-at(hi.x,lo.y-1);
}
@compute @workgroup_size(8,8)
fn distances(@builtin(global_invocation_id) id:vec3u) {
    if(id.x>=params.width || id.y>=params.height) {return;}
    if(params.cap==0u) {result[id.y*params.width+id.x]=0u;return;}
    // Most pixels are farther than the cap from a removed border. Certify
    // those in one rectangle query instead of searching every blank pixel.
    var lo=select(vec2u(0u),vec2u(params.cap),count(vec2i(id.xy),params.cap-1u)==vec2u(0u));
    var hi=vec2u(params.cap);
    loop {
        if(all(lo>=hi)) {break;}
        let mid=(lo+hi)/2u;
        let found=vec2u(count(vec2i(id.xy),mid.x).x,count(vec2i(id.xy),mid.y).y)>vec2u(0u);
        let searching=lo<hi;
        hi=select(hi,mid,vec2<bool>(searching.x&&found.x,searching.y&&found.y));
        lo=select(lo,mid+1u,vec2<bool>(searching.x&&(!found.x),searching.y&&(!found.y)));
    }
    result[id.y*params.width+id.x]=lo.x|(lo.y<<16u);
}
