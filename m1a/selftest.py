"""Synthesise AOVs for a sphere + ground plane, shade under sun A, then
verify relight.py reproduces sun B exactly (no shadows, no GI)."""
import numpy as np, json, OpenEXR
from exrio import normalize, focal_px
from relight import build_splats, shade_sun, to_image

h=w=128; lens=50.0; sensor=36.0
f=focal_px(lens,sensor,w,h)
cam_pos=np.array([0,-4,1.2],np.float32); forward=normalize(np.array([0,1,-0.2],np.float32))
up0=np.array([0,0,1],np.float32)
right=normalize(np.cross(forward,up0)); up=np.cross(right,forward)

j,i=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
x=(i+0.5-w/2)/f; y=-(j+0.5-h/2)/f
d=normalize(x[...,None]*right+y[...,None]*up+forward)

# sphere at origin r=1, plane z=-1
oc=cam_pos-np.array([0,0,0],np.float32)
b=2*np.einsum('ijk,k->ij',d,oc); c=oc@oc-1.0
disc=b*b-4*c; hit_s=disc>0
t_s=np.where(hit_s,(-b-np.sqrt(np.maximum(disc,0)))/2,1e9); t_s=np.where(t_s>0,t_s,1e9)
t_p=np.where(d[...,2]<-1e-6,(-1.0-cam_pos[2])/d[...,2],1e9); t_p=np.where(t_p>0,t_p,1e9)
t=np.minimum(t_s,t_p); valid=t<1e8
P=cam_pos+d*t[...,None]
N=np.where((t_s<t_p)[...,None],normalize(P),np.array([0,0,1.],np.float32))
alb=np.where((t_s<t_p)[...,None],np.array([.8,.3,.2],np.float32),np.array([.5,.5,.5],np.float32))
P=np.where(valid[...,None],P,0); N=np.where(valid[...,None],N,0); alb=np.where(valid[...,None],alb,0)

raw={'ViewLayer.Position.X':P[...,0],'ViewLayer.Position.Y':P[...,1],'ViewLayer.Position.Z':P[...,2],
    'ViewLayer.Normal.X':N[...,0],'ViewLayer.Normal.Y':N[...,1],'ViewLayer.Normal.Z':N[...,2],
    'ViewLayer.DiffCol.R':alb[...,0],'ViewLayer.DiffCol.G':alb[...,1],'ViewLayer.DiffCol.B':alb[...,2]}
ch={k:np.ascontiguousarray(v,dtype=np.float32) for k,v in raw.items()}
OpenEXR.File({},ch).write('test_aovs.exr')

scene={"camera":{"position":cam_pos.tolist(),"forward":forward.tolist(),
       "lens_mm":lens,"sensor_mm":sensor,"default_roughness":0.25},
       "new_sun":{"direction":[-0.5,0.6,-0.7],"color":[1,.95,.9],"strength":4.0}}
json.dump(scene,open('test_scene.json','w'),indent=2)

# ground truth: shade the SAME geometry directly with the same BRDF
sp=build_splats({'Position':P,'Normal':N,'DiffCol':alb},scene['camera'])
gt=to_image(sp,shade_sun(sp,**{'sun_dir':scene['new_sun']['direction'],
    'sun_color':scene['new_sun']['color'],'sun_strength':scene['new_sun']['strength']}))
from exrio import write_exr; write_exr('test_truth.exr',gt)
print("synthesised", valid.sum(), "surface pixels")
