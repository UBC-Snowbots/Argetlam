"""Derive rover2026 arm specs from the URDF + STL meshes (stdlib only)."""
import math, random, struct, json, glob, os, xml.etree.ElementTree as ET

URDF = r"C:\Users\prana\Desktop\UBC-Rover\LearnFlake\src\external_pkgs\rl_sar\src\robots\rover2026_description\urdf\rover2026.urdf"
MESH_GLOB = r"C:\Users\prana\Desktop\UBC-Rover\LearnFlake\src\external_pkgs\RoboSuite\robosuite\models\assets\robots\rover2026\meshes\*"
KEYS = [(0.11,-0.04,0.828),(0.05,-0.08,0.828),(0.07,-0.08,0.828),(0.09,-0.08,0.828),(0.05,-0.06,0.828),
        (0.07,-0.06,0.828),(0.09,-0.06,0.828),(0.05,-0.04,0.828),(0.07,-0.04,0.828),(0.09,-0.04,0.828)]
G = 9.81; N = 40000; PAYLOAD = 0.5  # kg assumed at EE-holder origin for the "with payload" column

def mm(A,B): return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def rpy(r,p,y):
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],[sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],[-sp, cp*sr, cp*cr]]
def axr(a,t):
    n=math.sqrt(sum(v*v for v in a)); x,y,z=(v/n for v in a); c,s=math.cos(t),math.sin(t); C=1-c
    return [[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]]
def H(R,p): return [R[0]+[p[0]],R[1]+[p[1]],R[2]+[p[2]],[0,0,0,1]]
def app(T,v): return [T[i][0]*v[0]+T[i][1]*v[1]+T[i][2]*v[2]+T[i][3] for i in range(3)]
def rot(T,v): return [T[i][0]*v[0]+T[i][1]*v[1]+T[i][2]*v[2] for i in range(3)]
def sub(a,b): return [a[i]-b[i] for i in range(3)]
def cross(a,b): return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def dot(a,b): return sum(a[i]*b[i] for i in range(3))
def norm(a): return math.sqrt(dot(a,a))

root=ET.parse(URDF).getroot()
links={}
for l in root.findall("link"):
    i=l.find("inertial"); o=i.find("origin")
    links[l.get("name")]={"mass":float(i.find("mass").get("value")),"com":[float(v) for v in o.get("xyz").split()]}
joints=[]
for j in root.findall("joint"):
    o=j.find("origin"); lim=j.find("limit")
    joints.append({"name":j.get("name"),"type":j.get("type"),"parent":j.find("parent").get("link"),
        "child":j.find("child").get("link"),"xyz":[float(v) for v in o.get("xyz").split()],
        "rpy":[float(v) for v in o.get("rpy").split()],"axis":[float(v) for v in j.find("axis").get("xyz").split()],
        "lo":float(lim.get("lower")),"hi":float(lim.get("upper"))})
for J in joints:  # continuous joint: treat as full rotation (URDF 0/0 limits are exporter junk)
    if J["type"]=="continuous": J["lo"],J["hi"]=-math.pi,math.pi

def fk(q):
    F={"base_link":[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}; jf=[]
    for J,qi in zip(joints,q):
        Tj=mm(F[J["parent"]],H(rpy(*J["rpy"]),J["xyz"]))  # joint frame (pre-rotation)
        jf.append((app(Tj,[0,0,0]),rot(Tj,J["axis"])))
        F[J["child"]]=mm(Tj,H(axr(J["axis"],qi),[0,0,0]))
    return F,jf

chain=[J["child"] for J in joints]
ee_link="a6_EE_holder"
# zero pose
F0,_=fk([0.0]*6)
ee0=app(F0[ee_link],[0,0,0]); sh0=app(F0["a1_shoulder_base"],[0,0,0])

random.seed(0)
lo=[J["lo"] for J in joints]; hi=[J["hi"] for J in joints]
mins=[1e9]*3; maxs=[-1e9]*3; maxreach=0; maxtau=[0]*6; maxtau_p=[0]*6; keymin=[1e9]*10
for _ in range(N):
    q=[random.uniform(lo[k],hi[k]) for k in range(6)]
    F,jf=fk(q); ee=app(F[ee_link],[0,0,0])
    for a in range(3): mins[a]=min(mins[a],ee[a]); maxs[a]=max(maxs[a],ee[a])
    maxreach=max(maxreach,norm(sub(ee,sh0)))
    for k,kp in enumerate(KEYS): keymin[k]=min(keymin[k],norm(sub(ee,kp)))
    coms=[(app(F[c],links[c]["com"]),links[c]["mass"]) for c in chain]
    for j,(p,a) in enumerate(jf):
        tau=sum(dot(a,cross(sub(c,p),[0,0,-m*G])) for c,m in coms[j:])
        maxtau[j]=max(maxtau[j],abs(tau))
        tp=tau+dot(a,cross(sub(ee,p),[0,0,-PAYLOAD*G]))
        maxtau_p[j]=max(maxtau_p[j],abs(tp))

# STL bounding boxes
def stl_bbox(path):
    with open(path,"rb") as f: data=f.read()
    n=struct.unpack("<I",data[80:84])[0]
    if 84+n*50==len(data):
        lo=[1e9]*3; hi=[-1e9]*3
        for t in range(n):
            base=84+t*50+12
            for v in range(3):
                p=struct.unpack("<3f",data[base+v*12:base+v*12+12])
                for a in range(3): lo[a]=min(lo[a],p[a]); hi[a]=max(hi[a],p[a])
        return [round((hi[a]-lo[a])*1000,1) for a in range(3)], n
    return None, 0
bbox={}
for p in glob.glob(MESH_GLOB):
    if p.lower().endswith(".stl"):
        b,n=stl_bbox(p); bbox[os.path.splitext(os.path.basename(p))[0]]={"bbox_mm":b,"tris":n}

out={
 "joints":[{"name":J["name"],"type":J["type"],"lo_deg":round(math.degrees(J["lo"]),1),"hi_deg":round(math.degrees(J["hi"]),1),
            "range_deg":round(math.degrees(J["hi"]-J["lo"]),1),"origin_offset_mm":round(norm(J["xyz"])*1000,1)} for J in joints],
 "links":{k:{"mass_kg":round(v["mass"],3)} for k,v in links.items()},
 "total_mass_kg":round(sum(v["mass"] for v in links.values()),3),
 "moving_mass_kg":round(sum(links[c]["mass"] for c in chain),3),
 "zero_pose_ee_m":[round(v,3) for v in ee0],"shoulder_origin_m":[round(v,3) for v in sh0],
 "workspace_bbox_m":{"min":[round(v,3) for v in mins],"max":[round(v,3) for v in maxs]},
 "max_reach_from_shoulder_m":round(maxreach,3),
 "key_min_dist_cm":[round(v*100,2) for v in keymin],
 "max_static_gravity_torque_Nm":{J["name"]:round(t,2) for J,t in zip(joints,maxtau)},
 f"max_static_torque_with_{PAYLOAD}kg_payload_Nm":{J["name"]:round(t,2) for J,t in zip(joints,maxtau_p)},
 "stl_bbox":bbox,"samples":N,
}
print(json.dumps(out,indent=1))
