struct Node { corners: array<vec2f,4>, offset:u32, mode:u32, padding:vec2u }
// sizes: of the source and of the output. grid: tiles per row, pixels per
// tile, neighbours per invocation, first output row of a pass over a band.
struct Params { sizes:vec4u, grid:vec4u }
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> nodes:array<Node>;
// The root node and the first exact coordinate of every tile, counted from
// the first ones of its band. The offsets in its nodes count from there.
@group(0) @binding(2) var<storage,read> roots:array<vec2u>;
@group(0) @binding(3) var<storage,read> dense:array<vec2f>;
@group(0) @binding(4) var<storage,read_write> image:array<u32>;
// One word per pixel and plane: a `vec2u` plane would double the largest
// binding and push full-resolution phone pages past the binding limit.
@group(0) @binding(5) var<storage,read_write> sums:array<u32>;
@group(0) @binding(6) var<storage,read_write> squares:array<u32>;
// Two neighbours of a row per word, each as gray and binary bytes.
@group(0) @binding(7) var<storage,read_write> tones:array<u32>;
@group(0) @binding(8) var<uniform> params:Params;

fn inverse(pixel:vec2u)->vec2f {
    var origin=(pixel/32u)*32u;
    var size=32u;
    let root=roots[(pixel.y/32u)*params.grid.x+pixel.x/32u];
    var index=root.x;
    for(var depth=0u;depth<6u;depth++) {
        let node=nodes[index];
        if(node.mode==2u) { return dense[root.y+node.offset+(pixel.y-origin.y)*size+pixel.x-origin.x]; }
        if(node.mode==0u) {
            let t=vec2f(pixel-origin)/f32(size);
            return mix(mix(node.corners[0],node.corners[1],t.x),mix(node.corners[2],node.corners[3],t.x),t.y);
        }
        size/=2u;
        let offset=vec2u(pixel>=origin+vec2u(size));
        origin+=offset*size;
        index=root.x+node.offset+offset.x+2u*offset.y;
    }
    return vec2f(-100.);
}
fn cubic(value:f32)->f32 {
    let x=abs(value);
    if(x<1.) {return 1.5*x*x*x-2.5*x*x+1.;}
    if(x<2.) {return -0.5*x*x*x+2.5*x*x-4.*x+2.;}
    return 0.;
}
fn source_rgb(p:vec2i)->vec3f {
    if(any(p<vec2i(0)) || any(p>=vec2i(params.sizes.xy))) {return vec3f(255.);}
    let word=source[u32(p.y)*params.sizes.x+u32(p.x)];
    return vec3f(f32(word&255u),f32((word>>8u)&255u),f32((word>>16u)&255u));
}
fn sample(p:vec2f)->vec3u {
    if(any(p!=p) || any(p<vec2f(-2.)) || any(p>vec2f(params.sizes.xy)+1.)) {return vec3u(255u);}
    let base=vec2i(floor(p));
    let wx=vec4f(cubic(p.x-f32(base.x-1)),cubic(p.x-f32(base.x)),cubic(p.x-f32(base.x+1)),cubic(p.x-f32(base.x+2)));
    let wy=vec4f(cubic(p.y-f32(base.y-1)),cubic(p.y-f32(base.y)),cubic(p.y-f32(base.y+1)),cubic(p.y-f32(base.y+2)));
    var value=vec3f(0.);
    for(var y=0;y<4;y++) {for(var x=0;x<4;x++) {
        value+=wx[x]*wy[y]*source_rgb(base+vec2i(x-1,y-1));
    }}
    return vec3u(clamp(floor(value+0.5),vec3f(0.),vec3f(255.)));
}
@compute @workgroup_size(8,8)
fn render(@builtin(global_invocation_id) id:vec3u) {
    let pixel=vec2u(id.x,id.y+params.grid.w);
    if(any(pixel>=params.sizes.zw)) {return;}
    let rgb=sample(inverse(pixel));
    let gray=(rgb.r*2126u+rgb.g*7152u+rgb.b*722u)/10000u;
    image[pixel.y*params.sizes.z+pixel.x]=rgb.r|(rgb.g<<8u)|(rgb.b<<16u)|(gray<<24u);
}
fn luminance(row:u32,x:i32)->vec2u {
    let v=image[row+u32(clamp(x,0,i32(params.sizes.z)-1))]>>24u;
    return vec2u(v,v*v);
}
// The window sums of `params.grid.z` neighbours in one invocation: all 51 taps
// for the first, then one tap in and one out for each further one.
@compute @workgroup_size(8,8)
fn horizontal(@builtin(global_invocation_id) id:vec3u) {
    let first=id.x*params.grid.z;
    let y=id.y+params.grid.w;
    if(first>=params.sizes.z || y>=params.sizes.w) {return;}
    let row=y*params.sizes.z;
    var sum=vec2u(0u);
    for(var x=-25;x<=25;x++) {sum+=luminance(row,i32(first)+x);}
    let end=min(first+params.grid.z,params.sizes.z);
    for(var x=first;x<end;x++) {
        sums[row+x]=sum.x;
        squares[row+x]=sum.y;
        sum+=luminance(row,i32(x)+26)-luminance(row,i32(x)-25);
    }
}
fn statistics(x:u32,y:i32)->vec2u {
    let index=u32(clamp(y,0,i32(params.sizes.w)-1))*params.sizes.z+x;
    return vec2u(sums[index],squares[index]);
}
fn tone(index:u32,sum:vec2u)->u32 {
    let mean=f32(sum.x)/2601.;
    let stddev=sqrt(max(f32(sum.y)/2601.-mean*mean,0.));
    let threshold=mean*(1.+0.25*(stddev/128.-1.));
    let gray=image[index]>>24u;
    let value=f32(gray);
    let binary=select(255u,0u,value<=threshold);
    // Continuous paper correction is shared CPU code after readback. The binary
    // plane is for diagnostics/content crop and must not gate gray coverage.
    return gray|(binary<<8u);
}
// Reads the sums of 25 rows above and below: one pass over the whole output.
@compute @workgroup_size(8,8)
fn paper(@builtin(global_invocation_id) id:vec3u) {
    let first=id.y*params.grid.z;
    // The two columns of a word; the last word of an odd width repeats its one.
    let x=vec2u(2u*id.x,min(2u*id.x+1u,params.sizes.z-1u));
    if(x[0]>=params.sizes.z || first>=params.sizes.w) {return;}
    var sum=array<vec2u,2>(vec2u(0u),vec2u(0u));
    for(var y=-25;y<=25;y++) {
        for(var k=0;k<2;k++) {sum[k]+=statistics(x[k],i32(first)+y);}
    }
    let end=min(first+params.grid.z,params.sizes.w);
    for(var y=first;y<end;y++) {
        let row=y*params.sizes.z;
        tones[y*((params.sizes.z+1u)/2u)+id.x]=tone(row+x[0],sum[0])|(tone(row+x[1],sum[1])<<16u);
        for(var k=0;k<2;k++) {sum[k]+=statistics(x[k],i32(y)+26)-statistics(x[k],i32(y)-25);}
    }
}
