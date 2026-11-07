struct Node { corners: array<vec2f,4>, offset:u32, mode:u32, padding:vec2u }
struct Params { sizes:vec4u, grid:vec4u }
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read> nodes:array<Node>;
@group(0) @binding(2) var<storage,read> roots:array<u32>;
@group(0) @binding(3) var<storage,read> dense:array<vec2f>;
@group(0) @binding(4) var<storage,read_write> image:array<u32>;
@group(0) @binding(5) var<storage,read_write> moments:array<vec2u>;
@group(0) @binding(6) var<storage,read_write> tones:array<u32>;
@group(0) @binding(7) var<uniform> params:Params;

fn inverse(pixel:vec2u)->vec2f {
    var origin=(pixel/32u)*32u;
    var size=32u;
    var index=roots[(pixel.y/32u)*params.grid.x+pixel.x/32u];
    for(var depth=0u;depth<6u;depth++) {
        let node=nodes[index];
        if(node.mode==2u) { return dense[node.offset+(pixel.y-origin.y)*size+pixel.x-origin.x]; }
        if(node.mode==0u) {
            let t=vec2f(pixel-origin)/f32(size);
            return mix(mix(node.corners[0],node.corners[1],t.x),mix(node.corners[2],node.corners[3],t.x),t.y);
        }
        size/=2u;
        let offset=vec2u(pixel>=origin+vec2u(size));
        origin+=offset*size;
        index=node.offset+offset.x+2u*offset.y;
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
    if(any(id.xy>=params.sizes.zw)) {return;}
    let rgb=sample(inverse(id.xy));
    let gray=(rgb.r*2126u+rgb.g*7152u+rgb.b*722u)/10000u;
    image[id.y*params.sizes.z+id.x]=rgb.r|(rgb.g<<8u)|(rgb.b<<16u)|(gray<<24u);
}
@compute @workgroup_size(8,8)
fn horizontal(@builtin(global_invocation_id) id:vec3u) {
    if(any(id.xy>=params.sizes.zw)) {return;}
    var sum=0u;var squares=0u;
    for(var x=-25;x<=25;x++) {
        let xx=u32(clamp(i32(id.x)+x,0,i32(params.sizes.z)-1));
        let v=image[id.y*params.sizes.z+xx]>>24u;
        sum+=v;squares+=v*v;
    }
    moments[id.y*params.sizes.z+id.x]=vec2u(sum,squares);
}
@compute @workgroup_size(8,8)
fn paper(@builtin(global_invocation_id) id:vec3u) {
    if(any(id.xy>=params.sizes.zw)) {return;}
    var sum=vec2u(0u);
    for(var y=-25;y<=25;y++) {
        let yy=u32(clamp(i32(id.y)+y,0,i32(params.sizes.w)-1));
        sum+=moments[yy*params.sizes.z+id.x];
    }
    let mean=f32(sum.x)/2601.;
    let stddev=sqrt(max(f32(sum.y)/2601.-mean*mean,0.));
    let threshold=mean*(1.+0.25*(stddev/128.-1.));
    let index=id.y*params.sizes.z+id.x;
    let gray=image[index]>>24u;
    let value=f32(gray);
    let binary=select(255u,0u,value<=threshold);
    let half=max(min(max(mean*0.1,stddev*0.5),mean*0.18),2.);
    let black=threshold-half;
    let white=min(threshold+half,mean*0.98);
    let clean=select(u32(floor(255.*clamp((value-black)/max(white-black,1.),0.,1.)+0.5)),0u,gray==0u);
    tones[index]=gray|(binary<<8u)|(clean<<16u)|0xff000000u;
}
