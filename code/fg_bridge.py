"""Closed-loop bridge to FlightGear 2020.3.19 with the ASA F-16CJ Block 52
(the exact simulator and aircraft the pilots flew; see the AsaFG README).

FlightGear runs its own JSBSim FDM, FBW and Nasal systems; Python talks to it
through FlightGear's property server (``--telnet``): it reads the ownship
state, computes the controls, and writes them back at 10 Hz of SIMULATED time.
The scripted leader is not simulated: relative position is computed against
the leader's logged track, exactly as in the training data.

Several instances can run side by side: set the environment variable
FG_INSTANCE=i (0, 1, ...) and every port is offset by 10*i.
"""
import os
import socket
import subprocess
import time

import numpy as np

FG_HOME = os.environ.get("FG_HOME", r"C:\Program Files\FlightGear 2020.3")
FGFS = os.path.join(FG_HOME, "bin", "fgfs.exe")
AIRCRAFT = "f16-block-52"
_INST = int(os.environ.get("FG_INSTANCE", "0"))
PORT = 5401 + 10 * _INST
R_FEET = 20902231.0
SLOT = np.array([-267.7, -11249.0, 193.4])  # leader-body-frame slot (ft): right, forward, up

STATE_PROPS = {
    "lat": "/position/latitude-deg",
    "lon": "/position/longitude-deg",
    "alt": "/position/altitude-ft",
    "roll": "/orientation/roll-deg",
    "pitch": "/orientation/pitch-deg",
    "heading": "/orientation/heading-deg",
    "kias": "/velocities/airspeed-kt",
    "vs_fps": "/velocities/vertical-speed-fps",
    "nz": "/accelerations/pilot/z-accel-fps_sec",
    "t": "/sim/time/elapsed-sec",
}
CONTROL_PROPS = ["/controls/flight/aileron", "/controls/flight/elevator", "/controls/flight/rudder",
                 "/controls/engines/engine[0]/throttle"]


class FGProps:
    """Minimal client for FlightGear's telnet property server (raw 'data' mode)."""

    def __init__(self, host="localhost", port=PORT, timeout=60.0):
        t0 = time.time()
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise
                time.sleep(1.0)
        self.sock.settimeout(300.0)  # the server answers only once FlightGear has finished loading
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # avoid Nagle/delayed-ACK stalls
        self.f = self.sock.makefile("rb")
        self._send("data")

    def wait_ready(self, timeout=600.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get("/sim/fdm-initialized").lower() in ("true", "1"):
                return True
            time.sleep(1.0)
        raise TimeoutError("FlightGear FDM not initialized")

    def _send(self, line):
        self.sock.sendall((line + "\r\n").encode())

    def get(self, prop):
        self._send(f"get {prop}")
        return self.f.readline().decode().strip()

    def getf(self, prop):
        v = self.get(prop)
        try:
            return float(v)
        except ValueError:
            return float("nan")

    def get_many(self, props):
        """Pipelined reads: all requests in one write, so the server answers
        them within a single frame instead of one frame per property."""
        self.sock.sendall("".join(f"get {p}\r\n" for p in props).encode())
        out = []
        for _ in props:
            v = self.f.readline().decode().strip()
            try:
                out.append(float(v))
            except ValueError:
                out.append(float("nan"))
        return out

    def set_many(self, pairs):
        self.sock.sendall("".join(f"set {p} {v}\r\n" for p, v in pairs).encode())

    def set(self, prop, value):
        self._send(f"set {prop} {value}")

    def run(self, command):
        self._send(f"run {command}")

    def close(self):
        try:
            self._send("quit")
        finally:
            self.sock.close()


# folder holding Protocol/wingman_{in,out}.xml (must be an ASCII path); override with FG_EXTRA_DATA
EXTRA_DATA = os.environ.get("FG_EXTRA_DATA", os.path.join(os.path.expanduser("~"), "fg_wingman_data"))
UDP_OUT, UDP_IN, UDP_HZ = 5501 + 10 * _INST, 5502 + 10 * _INST, 20


class FGUdp:
    """Fast I/O through FlightGear's generic protocol: state arrives as one CSV
    packet per frame of the output channel; controls go out as one packet."""
    FIELDS = ["t", "lat", "lon", "alt", "roll", "pitch", "heading", "kias", "vs_fps", "nz"]

    def __init__(self):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", UDP_OUT))
        self.rx.setblocking(False)
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def latest(self, timeout=10.0):
        """Most recent state packet (drains older ones); blocks until one arrives."""
        t0, last = time.time(), None
        while True:
            try:
                while True:
                    last = self.rx.recv(4096)
            except BlockingIOError:
                pass
            if last is not None:
                v = [float(x) for x in last.decode().strip().split(",")]
                return dict(zip(self.FIELDS, v))
            if time.time() - t0 > timeout:
                raise TimeoutError("no state packet from FlightGear")
            time.sleep(0.002)

    def send(self, u, trims=(0.0, 0.0)):
        """u = (aileron, elevator, rudder, throttle); trims = (aileron trim, elevator trim), held at 0 by
        every policy (the input protocol sets them explicitly so that nothing is left at a default)."""
        v = list(u)[:4] + list(trims)
        self.tx.sendto((",".join(f"{float(x):.5f}" for x in v) + "\n").encode(), ("127.0.0.1", UDP_IN))

    def close(self):
        self.rx.close()
        self.tx.close()


def launch(lat, lon, alt_ft, heading_deg, kias, roll_deg=0.0, pitch_deg=0.0, speed_up=1, extra=()):
    args = [FGFS, f"--fg-root={os.path.join(FG_HOME, 'data')}", f"--data={EXTRA_DATA}",
            f"--generic=socket,out,{UDP_HZ},127.0.0.1,{UDP_OUT},udp,wingman_out",
            f"--generic=socket,in,{UDP_HZ},127.0.0.1,{UDP_IN},udp,wingman_in",
            f"--aircraft={AIRCRAFT}", f"--lat={lat}", f"--lon={lon}", f"--altitude={alt_ft}",
            f"--heading={heading_deg}", f"--vc={kias}", f"--roll={roll_deg}", f"--pitch={pitch_deg}",
            "--in-air", "--state=cruise", f"--telnet={PORT}", "--disable-sound", "--disable-ai-traffic",
            "--disable-real-weather-fetch", "--disable-random-objects", "--timeofday=noon",
            "--geometry=640x480", f"--prop:/sim/speed-up={speed_up}", *extra]
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_state(fg):
    return dict(zip(STATE_PROPS.keys(), fg.get_many(list(STATE_PROPS.values()))))


def policy_inputs(st, leader):
    """leader = (lat, lon, alt_ft, heading_deg) at the same simulated time."""
    llat, llon, lalt, lhdg = leader
    dx = R_FEET * np.cos(np.radians(st["lat"])) * np.radians(st["lon"] - llon)
    dy = R_FEET * np.radians(st["lat"] - llat)
    dz = st["alt"] - lalt
    psi = np.radians(lhdg)
    xb, yb = dx * np.cos(psi) - dy * np.sin(psi), dx * np.sin(psi) + dy * np.cos(psi)
    rel_hdg = ((st["heading"] - lhdg + 180.0) % 360.0) - 180.0   # heading relative to the leader
    return np.array([xb, yb, dz, st["roll"], st["pitch"], rel_hdg, st["kias"], st["vs_fps"], st["nz"]])


class FormationController:
    """Cascade formation-keeping law used as the labeler for the flight domain.
    Lateral: slot offset -> heading correction -> bank -> aileron (roll-rate FBW).
    Vertical: altitude offset -> climb rate -> pitch -> elevator (+ = nose down).
    Longitudinal: along-track offset -> airspeed correction -> throttle.
    Gains are tuned in FlightGear before any policy is evaluated (validation step 1)."""

    def __init__(self, k_lat=0.003, max_dpsi=30.0, k_psi=2.0, max_bank=60.0, k_roll=0.04,
                 k_alt=0.05, max_vs=100.0, k_vs=0.02, k_pitch=0.15, k_lon=0.015, k_spd=0.02,
                 kias_ref=566.0, thr0=0.67):
        self.__dict__.update(locals())
        del self.__dict__["self"]

    def __call__(self, x, *_):
        """State feedback on the policy's own 9 inputs only (x[5] = heading relative
        to the leader), so a policy can represent it exactly. Accepts (n, 9) or (9,)."""
        X = np.atleast_2d(x)
        xb, yb, dz, roll, pitch, rel, kias, vs = (X[:, i] for i in range(8))
        e_lat, e_lon, e_alt = SLOT[0] - xb, SLOT[1] - yb, SLOT[2] - dz
        rel_cmd = np.clip(self.k_lat * e_lat, -self.max_dpsi, self.max_dpsi)
        bank_cmd = np.clip(self.k_psi * (rel_cmd - rel), -self.max_bank, self.max_bank)
        ail = np.clip(self.k_roll * (bank_cmd - roll), -1.0, 1.0)
        vs_cmd = np.clip(self.k_alt * e_alt, -self.max_vs, self.max_vs)
        pitch_cmd = np.clip(self.k_vs * (vs_cmd - vs), -10.0, 10.0)
        ele = np.clip(-self.k_pitch * (pitch_cmd - pitch), -1.0, 1.0)
        spd_cmd = self.kias_ref + np.clip(self.k_lon * e_lon, -120.0, 120.0)  # ahead of slot (e_lon<0) -> slower
        thr = np.clip(self.thr0 - self.k_spd * (kias - spd_cmd), 0.0, 1.0)
        out = np.column_stack([ail, ele, np.zeros(len(X)), thr])
        return out[0] if np.ndim(x) == 1 else out


def closed_loop(fg, act, leader_track, horizon_s=120.0, dt=0.1, controller_inputs=None):
    """Run `act` (maps the 9-D input to 4 controls) at `dt` of simulated time.
    leader_track: callable t -> (lat, lon, alt, heading, kias)."""
    t0 = fg.getf(STATE_PROPS["t"])
    next_t = t0
    log = []
    while True:
        st = read_state(fg)
        t = st["t"] - t0
        if t >= horizon_s:
            break
        if st["t"] < next_t:
            time.sleep(0.001)
            continue
        next_t += dt
        lead = leader_track(t)
        x = policy_inputs(st, lead[:4])
        u = act(x, lead) if controller_inputs else act(x)
        fg.set_many([(p, float(v)) for p, v in zip(CONTROL_PROPS, u)])
        log.append((t, *x, *u))
    return np.array(log)
