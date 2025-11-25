import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys

# Optimasi Tampilan
plt.style.use('fast')

print("=== DRONE SWARM V9 (HEAVY LIFT) ===")

# --- 1. INPUT ---
try:
    raw_load = input("1. Berat Beban (kg) [Default 10.0]: ")
    m_load = 10.0 if raw_load.strip() == "" else float(raw_load)
    if m_load <= 0.01: m_load = 0.1
    
    raw_drone = input("2. Jumlah Drone (unit) [Default 40]: ")
    num_drones = 40 if raw_drone.strip() == "" else int(raw_drone)
    if num_drones < 1: num_drones = 1

except ValueError:
    m_load = 10.0; num_drones = 40

# --- 2. GEOMETRI & FISIKA ---
# Jarak aman antar drone
SAFE_DISTANCE = 0.8 

# Hitung Radius Formasi
min_circumference = num_drones * SAFE_DISTANCE
min_radius = min_circumference / (2 * np.pi)
formation_radius = max(2.0, min_radius * 1.3) # Buffer 30% biar lega

# Hitung Panjang Tali Otomatis
# Agar tidak crash, tali harus lebih panjang dari hipotenusa formasi
# Kita set tinggi vertikal drone dari beban minimal 2.0 meter
min_vertical_h = 2.0
min_cable = np.sqrt(formation_radius**2 + min_vertical_h**2)
L_cable = max(3.0, min_cable) # Minimal panjang tali 3m

print(f"\n[SYSTEM SETUP]")
print(f"   -> Load per Drone : {m_load/num_drones:.2f} kg")
print(f"   -> Radius Formasi : {formation_radius:.2f} m")
print(f"   -> Panjang Tali   : {L_cable:.2f} m")

# TUNING STABILITAS (HEAVY DUTY)
# dt diperkecil karena 40 drone = 40 interaksi gaya yang kompleks
dt = 0.005         # Presisi tinggi (5ms)
k_cable = 150.0    # Kekakuan pegas
c_cable = 25.0     # Damping tinggi untuk meredam osilasi 40 drone
max_tension = 5000.0 # Batas tegangan tali dinaikkan

g = 9.81; m_drone = 1.0; 
total_time = 50.0 

# --- FORMASI ---
formation_offsets = []
for i in range(num_drones):
    angle = (2 * np.pi * i) / num_drones
    x = formation_radius * np.cos(angle)
    y = formation_radius * np.sin(angle)
    formation_offsets.append([x, y, 0.5]) # Z offset relative to payload
formation_offsets = np.array(formation_offsets)

# --- WAYPOINTS ---
waypoints = np.array([
    [0.0, 0.0, 0.0],    # Start
    [0.0, 0.0, 20.0],   # Angkat Tinggi (20m)
    [10.0, 5.0, 10.0],  # Geser Jauh
    [-5.0, -5.0, 5.0],  # Turun dan Geser
    [0.0, 0.0, 0.5]     # Landing
])
waypoint_radius = 2.5

# --- INITIAL STATE ---
p_load = np.array([0.0, 0.0, 0.0])
v_load = np.array([0.0, 0.0, 0.0])

# Start Position: TAUT (Tegang)
h_pyramid = np.sqrt(L_cable**2 - formation_radius**2)
p_drones = p_load + formation_offsets
p_drones[:, 2] = p_load[2] + h_pyramid 
v_drones = np.zeros((num_drones, 3))

# --- GAINS (POWER UP!) ---
# Kunci agar beban berat terangkat: KP Tinggi
kp = 4.0   # Power angkat (sebelumnya 3.0)
kd = 6.0    # Damping (agar tidak overshoot/mental)
REPULSION_GAIN = 15.0

history_drones = []; history_load = []; history_target = []

# --- PHYSICS ENGINE ---
def physics_step(p_d, v_d, p_l, v_l, u_forces, dt):
    total_tension = np.zeros(3)
    acc_drones = np.zeros_like(v_d)

    for i in range(num_drones):
        delta = p_l - p_d[i]; dist = np.linalg.norm(delta)
        dir_vec = delta / (dist + 1e-6) if dist > 1e-6 else np.zeros(3)

        tension = np.zeros(3)
        if dist > L_cable:
            stretch = dist - L_cable
            rel_vel = np.dot(v_l - v_d[i], dir_vec)
            # Heavy Duty Spring Logic
            f_mag = (k_cable * stretch) + (c_cable * rel_vel)
            f_mag = np.clip(f_mag, -100, max_tension) 
            tension = f_mag * dir_vec
        
        f_grav = np.array([0.0, 0.0, -m_drone * g])
        acc_drones[i] = (u_forces[i] + f_grav + tension) / m_drone
        total_tension -= tension

    f_grav_l = np.array([0.0, 0.0, -m_load * g])
    acc_load = (f_grav_l + total_tension) / m_load
    
    # Floor collision
    if p_l[2] < 0:
        p_l[2] = 0; v_l[2] = 0; acc_load[2] = max(0, acc_load[2])

    v_d += acc_drones * dt; p_d += v_d * dt
    v_l += acc_load * dt; p_l += v_l * dt
    
    if np.isnan(p_l).any(): return p_d, v_d, p_l, v_l # Safety return
        
    return p_d, v_d, p_l, v_l

def controller(p_d, v_d, p_l, target_load):
    u_forces = np.zeros((num_drones, 3))
    h_target = np.sqrt(L_cable**2 - formation_radius**2) 
    
    for i in range(num_drones):
        tgt_drone = target_load + formation_offsets[i]
        tgt_drone[2] += h_target
        
        err_pos = tgt_drone - p_d[i]
        
        # PID Standard
        f_nav = (kp * err_pos) - (kd * v_d[i])
        
        # Feedforward: Penting! Drone harus "tahu" dia menanggung beban
        # Setiap drone menanggung berat diri sendiri + (Berat Beban / Jumlah Drone)
        load_share = m_load / num_drones
        f_grav = np.array([0.0, 0.0, (m_drone * g) + (load_share * g)])
        
        # Repulsion (Anti-Tabrak)
        f_rep = np.zeros(3)
        for j in range(num_drones):
            if i == j: continue
            diff = p_d[i] - p_d[j]
            d_ij = np.linalg.norm(diff)
            if d_ij < SAFE_DISTANCE:
                mag = REPULSION_GAIN * (1.0/d_ij - 1.0/SAFE_DISTANCE)
                f_rep += mag * (diff/(d_ij+1e-6))
        
        u_forces[i] = f_nav + f_grav + f_rep
    return u_forces

# --- EXECUTION ---
print("\n[CALCULATING PHYSICS - HIGH PRECISION MODE]")
steps = int(total_time / dt)
sample_rate = int(0.05/dt) # Render di 20 FPS

for s in range(steps):
    if s % (steps // 10) == 0:
        print(f"Progress: {s/steps*100:.0f}%")

    tgt = waypoints[current_wp] if 'current_wp' in locals() else waypoints[0]
    if 'current_wp' not in locals(): current_wp = 0
    
    if np.linalg.norm(p_load - waypoints[current_wp]) < waypoint_radius and current_wp < len(waypoints)-1:
        current_wp += 1
        tgt = waypoints[current_wp]

    u = controller(p_drones, v_drones, p_load, tgt)
    p_drones, v_drones, p_load, v_load = physics_step(p_drones, v_drones, p_load, v_load, u, dt)
    
    if np.isnan(p_load).any():
        print("⚠ ERROR: Crash detected.")
        break

    if s % sample_rate == 0:
        history_drones.append(p_drones.copy())
        history_load.append(p_load.copy())
        history_target.append(tgt.copy())

print("Progress: 100% - Done!")

# --- VISUALIZATION ---
print("[RENDERING ANIMATION]")
hist_d = np.array(history_drones); hist_l = np.array(history_load); hist_t = np.array(history_target)

if len(hist_l) == 0: sys.exit()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Axis limit dinamis (Area Luas)
limit = max(15, formation_radius + 10)
ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(0, 20)
ax.set_title(f"Heavy Lift Swarm: {num_drones} Drones | {m_load} kg Load")

# Tentukan ukuran marker berdasarkan jumlah drone (biar ga sumpek)
drone_ms = 4 if num_drones < 20 else 2

ln_cables = [ax.plot([],[],[],'k-',lw=0.3)[0] for _ in range(num_drones)]
pt_drones = [ax.plot([],[],[],'bo',ms=drone_ms)[0] for _ in range(num_drones)]
pt_load, = ax.plot([],[],[],'ro',ms=6, label='Payload')
pt_tgt, = ax.plot([],[],[],'gx',ms=8, label='Target')
ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'g--', alpha=0.3)

def update(f):
    if f >= len(hist_l): return
    d = hist_d[f]; l = hist_l[f]; t = hist_t[f]
    
    pt_load.set_data([l[0]],[l[1]]); pt_load.set_3d_properties([l[2]])
    pt_tgt.set_data([t[0]],[t[1]]); pt_tgt.set_3d_properties([t[2]])
    
    for i in range(num_drones):
        pt_drones[i].set_data([d[i,0]],[d[i,1]]); pt_drones[i].set_3d_properties([d[i,2]])
        ln_cables[i].set_data([d[i,0],l[0]],[d[i,1],l[1]]); ln_cables[i].set_3d_properties([d[i,2],l[2]])
        
    return pt_drones + ln_cables + [pt_load, pt_tgt]

ani = animation.FuncAnimation(fig, update, frames=len(hist_l), interval=30, blit=False)
plt.legend()
plt.show()