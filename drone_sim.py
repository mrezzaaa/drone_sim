import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- KONFIGURASI FISIKA ---
g = 9.81           
m_drone = 1.0      
m_load = 0.5       
L = 1.0            
dt = 0.02          
total_time = 20.0  # Durasi diperpanjang untuk selesaikan rute

# --- KONFIGURASI WAYPOINTS (X, Y) ---
waypoints = np.array([
    [0.0, 0.0],   # Start
    [1.0, 2.0],   # Titik 1 (Naik)
    [3.0, 2.0],   # Titik 2 (Geser Kanan)
    [3.0, 0.0],   # Titik 3 (Turun)
    [0.0, 0.0]    # Titik 4 (Kembali ke Awal)
])
waypoint_radius = 0.2 # Jarak toleransi untuk ganti target

# --- INITIAL STATE ---
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# --- CONTROLLER GAINS ---
kp_pos = 4.0   
kd_pos = 3.0   
k_swing = 12.0 

history_drone = []
history_load = []
history_time = []
active_target_history = [] # Untuk visualisasi target yang aktif

def physics_step(state, u_x, u_y, dt):
    x, vx, y, vy, theta, omega = state

    # Coupling force
    tension_x = m_load * g * np.sin(theta) * 0.5 
    tension_y = m_load * g * np.cos(theta) 

    ax = (u_x + tension_x) / m_drone
    ay = (u_y - m_drone * g - tension_y) / m_drone

    # Pendulum dynamics with damping
    alpha = (-g * np.sin(theta) - ax * np.cos(theta) - ay * np.sin(theta)) / L - (0.1 * omega)

    vx_new = vx + ax * dt
    x_new = x + vx_new * dt
    vy_new = vy + ay * dt
    y_new = y + vy_new * dt
    omega_new = omega + alpha * dt
    theta_new = theta + omega_new * dt

    return np.array([x_new, vx_new, y_new, vy_new, theta_new, omega_new])

def controller(state, target):
    x, vx, y, vy, theta, omega = state
    
    ex = target[0] - x
    ey = target[1] - y

    fx_des = kp_pos * ex - kd_pos * vx
    fy_des = kp_pos * ey - kd_pos * vy

    swing_correction = k_swing * np.sin(theta)
    
    u_x = fx_des + swing_correction 
    u_y = fy_des + (m_drone + m_load) * g 

    return u_x, u_y

# --- SIMULATION LOOP ---
current_state = np.copy(state)
current_wp_idx = 0 # Indeks waypoint saat ini

for t in np.arange(0, total_time, dt):
    # 1. Logic Ganti Waypoint
    target = waypoints[current_wp_idx]
    
    # Hitung jarak drone ke target saat ini
    dist_to_target = np.linalg.norm(current_state[[0, 2]] - target)
    
    # Jika sudah dekat DAN bukan titik terakhir, pindah ke titik berikutnya
    if dist_to_target < waypoint_radius and current_wp_idx < len(waypoints) - 1:
        current_wp_idx += 1
        target = waypoints[current_wp_idx]

    # 2. Hitung Control & Fisika
    ux, uy = controller(current_state, target)
    current_state = physics_step(current_state, ux, uy, dt)
    
    # 3. Simpan Data
    dx, _, dy, _, th, _ = current_state
    lx = dx + L * np.sin(th)
    ly = dy - L * np.cos(th)
    
    history_drone.append([dx, dy])
    history_load.append([lx, ly])
    history_time.append(t)
    active_target_history.append(target)

# --- VISUALIZATION ---
fig, ax = plt.subplots()
ax.set_xlim(-1, 5)
ax.set_ylim(-2, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Drone Waypoint Navigation with Swing Suppression")

# Plot jalur statis (Rencana Terbang)
ax.plot(waypoints[:, 0], waypoints[:, 1], 'g--', alpha=0.5, label='Flight Path')

line_cable, = ax.plot([], [], 'k-', lw=1)
point_drone, = ax.plot([], [], 'bo', ms=10, label='Drone')
point_load, = ax.plot([], [], 'ro', ms=8, label='Load') # Fixed: 'Ro' -> 'ro'
target_marker, = ax.plot([], [], 'gx', ms=12, markeredgewidth=2, label='Active Target')

def init():
    line_cable.set_data([], [])
    point_drone.set_data([], [])
    point_load.set_data([], [])
    target_marker.set_data([], [])
    return line_cable, point_drone, point_load, target_marker

def animate(i):
    dx, dy = history_drone[i]
    lx, ly = history_load[i]
    tx, ty = active_target_history[i] # Posisi target pada saat t
    
    line_cable.set_data([dx, lx], [dy, ly])
    point_drone.set_data([dx], [dy])
    point_load.set_data([lx], [ly])
    target_marker.set_data([tx], [ty])
    
    return line_cable, point_drone, point_load, target_marker

ani = animation.FuncAnimation(fig, animate, frames=len(history_time),
                              init_func=init, interval=dt*1000, blit=True)

plt.legend(loc='upper right')
plt.show()
