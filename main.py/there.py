import tkinter as tk
import heapq
import random
from math import inf

# ---------- Pathfinding (A*) ----------
def astar(grid, start, goal, cost_fn=None):
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def passable(r, c): return grid[r][c] == 0

    def heuristic(r, c):
        return abs(r - gr) + abs(c - gc)  # Manhattan

    def step_cost(r, c):
        base = 1.0
        extra = 0.0 if cost_fn is None else float(cost_fn(r, c))
        return base + extra

    frontier = []
    heapq.heappush(frontier, (heuristic(sr, sc), 0.0, (sr, sc)))

    came_from = { (sr, sc): None }
    g_score = { (sr, sc): 0.0 }

    neighbors = [(1,0), (-1,0), (0,1), (0,-1)]

    while frontier:
        _, g, (r, c) = heapq.heappop(frontier)

        if (r, c) == (gr, gc):
            path = []
            cur = (r, c)
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path

        # Skip stale queue entries
        if g > g_score.get((r, c), inf):
            continue

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc) or not passable(nr, nc):
                continue

            tentative_g = g + step_cost(nr, nc)
            if tentative_g < g_score.get((nr, nc), inf):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = (r, c)
                f = tentative_g + heuristic(nr, nc)
                heapq.heappush(frontier, (f, tentative_g, (nr, nc)))

    return None


# ---------- Random "flood blobs" generator ----------
def add_flood_blobs(depth, n_blobs, blob_radius, blob_strength, seed=None):
    """
    depth: 2D float matrix; mutated in place
    Makes soft blobs: higher numbers = more "flood" cost.
    """
    if seed is not None:
        random.seed(seed)

    rows, cols = len(depth), len(depth[0])

    for _ in range(n_blobs):
        cr = random.randint(1, rows - 2)
        cc = random.randint(1, cols - 2)
        rad = random.randint(max(1, blob_radius - 2), blob_radius + 2)
        strength = random.uniform(blob_strength * 0.6, blob_strength * 1.2)

        for r in range(cr - rad, cr + rad + 1):
            if r < 1 or r >= rows - 1:
                continue
            for c in range(cc - rad, cc + rad + 1):
                if c < 1 or c >= cols - 1:
                    continue
                # Cheap circular-ish falloff
                dist = abs(r - cr) + abs(c - cc)
                if dist > rad:
                    continue
                add = strength * (1.0 - dist / (rad + 1e-9))
                depth[r][c] += max(0.0, add)


# ---------- GUI ----------
class AStarGUI:
    def __init__(self, rows=30, cols=45, cell_size=18):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        self.root = tk.Tk()
        self.root.title("A* Pathfinder (Shift=Start, Ctrl=Goal, Click=Walls)")

        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]   # 0 free, 1 wall
        self.flood = [[0.0 for _ in range(cols)] for _ in range(rows)]  # >=0 flood depth/cost
        self.start = (1, 1)
        self.goal = (rows - 2, cols - 2)

        self.path_cells = set()

        # Controls
        top = tk.Frame(self.root)
        top.pack(fill="x")

        tk.Button(top, text="Run A*", command=self.run_astar).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Clear Path", command=self.clear_path).pack(side="left", padx=4, pady=4)

        tk.Button(top, text="Random Walls", command=self.random_walls).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Random Flood Blobs", command=self.random_flood).pack(side="left", padx=4, pady=4)
        tk.Button(top, text="Clear Flood", command=self.clear_flood).pack(side="left", padx=4, pady=4)

        tk.Button(top, text="Reset All", command=self.reset_all).pack(side="left", padx=4, pady=4)

        # Sliders (simple, fun)
        self.wall_density = tk.DoubleVar(value=0.25)
        self.blob_count = tk.IntVar(value=10)
        self.blob_radius = tk.IntVar(value=7)
        self.blob_strength = tk.DoubleVar(value=1.8)
        self.unsafe_depth = tk.DoubleVar(value=4.0)   # flood >= this becomes "blocked"
        self.penalty_scale = tk.DoubleVar(value=1.2)  # flood cost multiplier

        sliders = tk.Frame(self.root)
        sliders.pack(fill="x")

        self._slider(sliders, "Wall density", self.wall_density, 0.05, 0.60, 0.01)
        self._slider(sliders, "Blob count", self.blob_count, 0, 40, 1, is_int=True)
        self._slider(sliders, "Blob radius", self.blob_radius, 2, 20, 1, is_int=True)
        self._slider(sliders, "Blob strength", self.blob_strength, 0.2, 5.0, 0.1)
        self._slider(sliders, "Unsafe depth", self.unsafe_depth, 0.5, 10.0, 0.5)
        self._slider(sliders, "Penalty scale", self.penalty_scale, 0.0, 5.0, 0.1)

        self.status = tk.StringVar(value="Click=wall | Shift+Click=start | Ctrl+Click=goal | Borders are walls.")
        tk.Label(self.root, textvariable=self.status, anchor="w").pack(fill="x", padx=8, pady=(0,6))

        # Canvas
        w = cols * cell_size
        h = rows * cell_size
        self.canvas = tk.Canvas(self.root, width=w, height=h)
        self.canvas.pack()

        self.rects = [[None for _ in range(cols)] for _ in range(rows)]
        self.draw_grid()

        # Bind clicks
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Shift-Button-1>", self.on_shift_left_click)
        self.canvas.bind("<Control-Button-1>", self.on_ctrl_left_click)

        # Set border walls initially
        self.apply_border_walls()
        self.redraw_all()

    def _slider(self, parent, label, var, mn, mx, step, is_int=False):
        box = tk.Frame(parent)
        box.pack(side="left", padx=6)
        tk.Label(box, text=label).pack()
        if is_int:
            s = tk.Scale(box, from_=mn, to=mx, orient="horizontal", resolution=step,
                         variable=var, length=140)
        else:
            s = tk.Scale(box, from_=mn, to=mx, orient="horizontal", resolution=step,
                         variable=var, length=140)
        s.pack()

    def draw_grid(self):
        cs = self.cell_size
        for r in range(self.rows):
            for c in range(self.cols):
                x1, y1 = c * cs, r * cs
                x2, y2 = x1 + cs, y1 + cs
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="gray")
                self.rects[r][c] = rect

    def cell_from_event(self, event):
        cs = self.cell_size
        c = event.x // cs
        r = event.y // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return None

    def set_cell_color(self, r, c, color):
        self.canvas.itemconfig(self.rects[r][c], fill=color)

    def apply_border_walls(self):
        for r in range(self.rows):
            self.grid[r][0] = 1
            self.grid[r][self.cols - 1] = 1
        for c in range(self.cols):
            self.grid[0][c] = 1
            self.grid[self.rows - 1][c] = 1

        # Ensure start/goal aren't walled
        sr, sc = self.start
        gr, gc = self.goal
        self.grid[sr][sc] = 0
        self.grid[gr][gc] = 0

    def clear_path(self):
        self.path_cells.clear()
        self.redraw_all()
        self.status.set("Path cleared.")

    def clear_flood(self):
        for r in range(self.rows):
            for c in range(self.cols):
                self.flood[r][c] = 0.0
        self.path_cells.clear()
        self.redraw_all()
        self.status.set("Flood cleared.")

    def reset_all(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.flood = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self.path_cells.clear()
        self.apply_border_walls()
        self.redraw_all()
        self.status.set("Reset done (border walls on).")

    def toggle_wall(self, r, c):
        # prevent editing border walls (keeps your “no edge hugging” promise)
        if r == 0 or c == 0 or r == self.rows - 1 or c == self.cols - 1:
            return
        if (r, c) == self.start or (r, c) == self.goal:
            return
        self.grid[r][c] = 0 if self.grid[r][c] == 1 else 1

    def on_left_click(self, event):
        cell = self.cell_from_event(event)
        if not cell:
            return
        r, c = cell
        self.toggle_wall(r, c)
        self.path_cells.clear()
        self.redraw_all()

    def on_shift_left_click(self, event):
        cell = self.cell_from_event(event)
        if not cell:
            return
        r, c = cell
        if self.grid[r][c] == 1:
            return
        self.start = (r, c)
        if self.start == self.goal:
            self.goal = (self.rows - 2, self.cols - 2)
        self.grid[self.start[0]][self.start[1]] = 0
        self.path_cells.clear()
        self.redraw_all()
        self.status.set(f"Start set to {self.start}")

    def on_ctrl_left_click(self, event):
        cell = self.cell_from_event(event)
        if not cell:
            return
        r, c = cell
        if self.grid[r][c] == 1:
            return
        self.goal = (r, c)
        if self.goal == self.start:
            self.start = (1, 1)
        self.grid[self.goal[0]][self.goal[1]] = 0
        self.path_cells.clear()
        self.redraw_all()
        self.status.set(f"Goal set to {self.goal}")

    def random_walls(self):
        d = float(self.wall_density.get())

        # Keep border walls. Randomize interior.
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if (r, c) == self.start or (r, c) == self.goal:
                    self.grid[r][c] = 0
                    continue
                # Do not place walls on top of "very deep flood" only; walls independent
                self.grid[r][c] = 1 if random.random() < d else 0

        self.apply_border_walls()
        self.path_cells.clear()
        self.redraw_all()
        self.status.set(f"Random walls generated (density={d:.2f}).")

    def random_flood(self):
        # Add some blobs onto existing flood
        n = int(self.blob_count.get())
        rad = int(self.blob_radius.get())
        strength = float(self.blob_strength.get())

        add_flood_blobs(self.flood, n_blobs=n, blob_radius=rad, blob_strength=strength)
        self.path_cells.clear()
        self.redraw_all()
        self.status.set(f"Added flood blobs (count={n}, radius≈{rad}).")

    def flood_cost_fn(self, r, c):
        d = float(self.flood[r][c])
        unsafe = float(self.unsafe_depth.get())
        scale = float(self.penalty_scale.get())

        # Treat very deep flood as basically blocked
        if d >= unsafe:
            return 1e9
        return d * scale

    def redraw_all(self):
        # Base cells: walls vs free
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    self.set_cell_color(r, c, "black")
                else:
                    # Flood shading: more flood => darker blue-ish
                    # We keep it simple with a few bands.
                    d = self.flood[r][c]
                    if d <= 0.01:
                        self.set_cell_color(r, c, "white")
                    elif d < 1.0:
                        self.set_cell_color(r, c, "lightcyan")
                    elif d < 2.5:
                        self.set_cell_color(r, c, "paleturquoise")
                    elif d < float(self.unsafe_depth.get()):
                        self.set_cell_color(r, c, "turquoise")
                    else:
                        # Unsafe depth still drawn as flood color, but A* treats it as blocked cost
                        self.set_cell_color(r, c, "deepskyblue")

        # Path overlay
        for (r, c) in self.path_cells:
            if (r, c) != self.start and (r, c) != self.goal:
                self.set_cell_color(r, c, "gold")

        # Start/Goal overlay
        sr, sc = self.start
        gr, gc = self.goal
        self.set_cell_color(sr, sc, "limegreen")
        self.set_cell_color(gr, gc, "tomato")

    def run_astar(self):
        self.path_cells.clear()

        sr, sc = self.start
        gr, gc = self.goal
        if self.grid[sr][sc] == 1 or self.grid[gr][gc] == 1:
            self.status.set("Start/Goal is blocked. Unblock it first.")
            return

        path = astar(self.grid, self.start, self.goal, cost_fn=self.flood_cost_fn)
        if path is None:
            self.status.set("No path found 😭 (walls/flood too cursed?)")
        else:
            self.path_cells = set(path)
            self.status.set(
                f"Path length: {len(path)} | "
                f"unsafe_depth={float(self.unsafe_depth.get()):.1f}, "
                f"penalty_scale={float(self.penalty_scale.get()):.1f}"
            )
        self.redraw_all()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = AStarGUI(rows=30, cols=45, cell_size=18)
    app.run()