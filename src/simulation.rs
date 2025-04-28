use rayon::prelude::*;
use std::f32::consts::PI;

use crate::vectors::Vector2D;
use crate::vectors::VectorDirecitons;

pub const INITIAL_DOT_COUNT: usize = 10000;
pub const INITIAL_GRAVITY: f32 = 100.; // Adjusted gravity might be needed
pub const INITIAL_DOT_RADIUS: f32 = 2.0;
const SMOOTHING_RADIUS: f32 = 16.0;

// Choose cell size based on smoothing radius
const GRID_CELL_SIZE: f32 = SMOOTHING_RADIUS; // Or slightly larger

// SPH Constants
const PARTICLE_MASS: f32 = 1.0;
const TARGET_DENSITY: f32 = 0.02;
const PRESSURE_MULTIPLIER: f32 = 4.0;
const EOS_GAMMA: f32 = 7.0;
const VISCOSITY_STRENGTH: f32 = 2.; // Added back - adjust value as needed
const BOUNDARY_DAMPING: f32 = 0.01; // Damping factor for boundary collisions
// Visualization Constants
const MAX_EXPECTED_DENSITY: f32 = 0.01; // Tune this based on observed densities
const DENSITY_COLOR: [u8; 3] = [50, 80, 150]; // Blueish color for density
const BYTES_PER_PIXEL: usize = 4;

pub struct Simulation {
    dot_radius: f32,
    dot_positions: Vec<Vector2D>,
    dot_velocities: Vec<Vector2D>,
    dot_densities: Vec<f32>,
    pub width: f32,
    pub height: f32,
    gravity: f32,
    background_color: [u8; 4],
    particle_count: usize,
    // Spatial Grid fields
    grid_cells: Vec<Vec<usize>>, // Stores particle indices for each cell
    grid_width: usize,
    grid_height: usize,
}

impl Default for Simulation {
    fn default() -> Self {
        Self {
            dot_positions: vec![],
            dot_velocities: vec![],
            dot_densities: vec![],
            dot_radius: INITIAL_DOT_RADIUS,
            width: 0.,
            height: 0.,
            gravity: INITIAL_GRAVITY,
            background_color: [24, 24, 24, 255],
            particle_count: 0,
            // Initialize grid fields (will be properly sized later)
            grid_cells: vec![],
            grid_width: 0,
            grid_height: 0,
        }
    }
}

fn smoothing_kernel(radius: f32, distance: f32) -> f32 {
    if distance >= radius {
        return 0.;
    }

    let volume = PI * f32::powi(radius, 4) / 6.;
    return (radius - distance) * (radius - distance) / volume;
}

fn smoothing_kernel_derivative(radius: f32, distance: f32) -> f32 {
    if distance >= radius || distance < 1e-6 {
        // Avoid division by zero/sqrt(0)
        return 0.;
    }
    // Based on Spiky Kernel: -(radius - distance)^2 / (PI * radius^6 / 6) * (1/distance) * gradient_factor?
    // Simpler Poly6-like derivative used before:
    let scale = 12. / (PI * f32::powi(radius, 4)); // From Poly6 gradient calculation factor
    return (distance - radius) * scale; // This provides direction via (r-r_i) later
}

impl Simulation {
    // Helper to get grid cell coordinates from world position
    fn get_cell_coords(&self, position: Vector2D) -> (isize, isize) {
        (
            (position[0] / GRID_CELL_SIZE) as isize,
            (position[1] / GRID_CELL_SIZE) as isize,
        )
    }

    // Helper to get flattened grid index from cell coordinates
    fn get_cell_index(&self, cx: isize, cy: isize) -> Option<usize> {
        if cx >= 0 && cx < self.grid_width as isize && cy >= 0 && cy < self.grid_height as isize {
            Some((cy * self.grid_width as isize + cx) as usize)
        } else {
            None // Out of bounds
        }
    }

    // Function to initialize or resize the grid
    fn initialize_grid(&mut self) {
        self.grid_width = (self.width / GRID_CELL_SIZE).ceil() as usize;
        self.grid_height = (self.height / GRID_CELL_SIZE).ceil() as usize;
        let num_cells = self.grid_width * self.grid_height;
        // Optimization: only reallocate if size changed drastically?
        // For simplicity, we clear and resize each time for now.
        self.grid_cells = vec![Vec::with_capacity(10); num_cells]; // Pre-allocate inner Vecs slightly
        println!(
            "Grid initialized: {}x{} cells",
            self.grid_width, self.grid_height
        );
    }

    // Function to clear and rebuild the grid with current particle positions
    fn update_grid(&mut self) {
        // Ensure grid is correctly sized (call after width/height are set)
        if self.grid_cells.is_empty() || self.grid_cells.len() != self.grid_width * self.grid_height
        {
            self.initialize_grid();
        }

        // Clear previous frame's particle indices from cells
        for cell in self.grid_cells.iter_mut() {
            cell.clear();
        }

        // Populate grid with current particle indices
        for i in 0..self.particle_count {
            let pos = self.dot_positions[i];
            let (cx, cy) = self.get_cell_coords(pos);
            if let Some(index) = self.get_cell_index(cx, cy) {
                // Bounds check just in case
                if index < self.grid_cells.len() {
                    self.grid_cells[index].push(i);
                }
            }
        }
    }

    pub fn add_dot(&mut self, position: Vector2D, velocity: Vector2D) {
        self.dot_positions.push(position);
        self.dot_velocities.push(velocity);
        self.dot_densities.push(0.);
        self.particle_count += 1;
        // Grid will be updated in `update`
    }

    pub fn update(&mut self, time_delta: f32) {
        // --- Update Grid --- (Needs to happen before density/force calcs)
        self.update_grid();

        // --- Calculate Densities --- (Parallel)
        let densities: Vec<f32> = (0..self.particle_count)
            .into_par_iter()
            .map(|i| self.calculate_particle_density(i))
            .collect();
        self.dot_densities = densities; // Update simulation state

        // --- Calculate Interaction Forces --- (Parallel)
        // Need immutable references to pass to the parallel closure
        let positions_ref = &self.dot_positions;
        let velocities_ref = &self.dot_velocities; // Add velocities back
        let densities_ref = &self.dot_densities;
        let grid_cells_ref = &self.grid_cells;
        let grid_w = self.grid_width;
        let grid_h = self.grid_height;

        let forces: Vec<Vector2D> = (0..self.particle_count)
            .into_par_iter()
            .map(|i| {
                calculate_interaction_forces(
                    i,
                    positions_ref,
                    velocities_ref, // Pass velocities
                    densities_ref,
                    grid_cells_ref,
                    grid_w,
                    grid_h,
                )
            })
            .collect();

        // --- Apply Forces and Update Physics --- (Parallel)
        let gravity_accel = Vector2D::DOWN * Vector2D::splat(self.gravity);
        let h = self.height;
        let r = self.dot_radius;
        // let dt = Vector2D::splat(time_delta); // USE actual time delta
        let dt = Vector2D::splat(1.0 / 56.0);
        self.dot_positions
            .par_iter_mut()
            .zip(self.dot_velocities.par_iter_mut())
            .zip(forces.par_iter()) // Add forces iterator
            .zip(self.dot_densities.par_iter()) // Add densities iterator
            .for_each(|(((pos, vel), force), density)| {
                // Calculate acceleration from force: a = F / rho
                let pressure_accel = if *density > 1e-6 {
                    *force / Vector2D::splat(*density)
                } else {
                    Vector2D::default()
                };

                // Update velocity
                *vel += (gravity_accel + pressure_accel) * dt;

                // Simple velocity clamping (optional)
                const MAX_VEL: f32 = 1000.0; // This might need adjustment too
                let vel_mag_sq = vel.length_squared();
                if vel_mag_sq > MAX_VEL * MAX_VEL {
                    *vel = *vel * Vector2D::splat(MAX_VEL / vel_mag_sq.sqrt());
                }

                // Update position
                *pos += *vel * dt;

                // Boundary conditions (collision)
                if (*pos)[1] > h - r {
                    *vel = Vector2D::from_array([(*vel)[0], -(*vel)[1] * BOUNDARY_DAMPING]); // Use damping constant
                    *pos = Vector2D::from_array([(*pos)[0], h - r]);
                }
                // Add other boundaries (left, right, top?)
                if (*pos)[0] < r {
                    *vel = Vector2D::from_array([-(*vel)[0] * BOUNDARY_DAMPING, (*vel)[1]]); // Use damping constant
                    *pos = Vector2D::from_array([r, (*pos)[1]]);
                }
                if (*pos)[0] > self.width - r {
                    *vel = Vector2D::from_array([-(*vel)[0] * BOUNDARY_DAMPING, (*vel)[1]]); // Use damping constant
                    *pos = Vector2D::from_array([self.width - r, (*pos)[1]]);
                }
                if (*pos)[1] < r {
                    // Ceiling collision
                    *vel = Vector2D::from_array([(*vel)[0], -(*vel)[1] * BOUNDARY_DAMPING]); // Use damping constant
                    *pos = Vector2D::from_array([(*pos)[0], r]);
                }
            });

        // --- Density visualization update (Optional) ---
        // self.dot_densities is already updated if you want to use it directly
    }

    // Calculate density for a single particle using the grid
    fn calculate_particle_density(&self, particle_index: usize) -> f32 {
        let mut density = 0.;
        let pos_i = self.dot_positions[particle_index];
        let (scx, scy) = self.get_cell_coords(pos_i);

        // Iterate over the 3x3 grid neighborhood
        for ny in -1..=1 {
            for nx in -1..=1 {
                let cx = scx + nx;
                let cy = scy + ny;

                if let Some(cell_index) = self.get_cell_index(cx, cy) {
                    // No need for: if cell_index < self.grid_cells.len()
                    for neighbor_index in &self.grid_cells[cell_index] {
                        // Density contribution includes self later, so no need for i == j check here
                        let pos_j = self.dot_positions[*neighbor_index];
                        let diff = pos_i - pos_j;
                        let dist_sq = diff.length_squared();

                        if dist_sq < SMOOTHING_RADIUS * SMOOTHING_RADIUS {
                            let distance = dist_sq.sqrt();
                            density += PARTICLE_MASS * smoothing_kernel(SMOOTHING_RADIUS, distance);
                        }
                    }
                }
            }
        }

        density
    }

    // Calculate density at an arbitrary sample point for visualization
    fn calculate_density_for_visualisation(&self, sample_point: Vector2D) -> f32 {
        let mut density = 0.;
        let (scx, scy) = self.get_cell_coords(sample_point);

        // Iterate over the 3x3 grid neighborhood around the sample point's cell
        for ny in -1..=1 {
            for nx in -1..=1 {
                let cx = scx + nx;
                let cy = scy + ny;

                if let Some(cell_index) = self.get_cell_index(cx, cy) {
                    // Check particles within this neighboring cell
                    // No need for bounds check on cell_index as get_cell_index handles it
                    for particle_index in &self.grid_cells[cell_index] {
                        // Check if particle_index is valid for dot_positions lookup
                        if *particle_index < self.dot_positions.len() {
                            let position = self.dot_positions[*particle_index];
                            let diff = position - sample_point;
                            let dist_sq = diff.length_squared();

                            if dist_sq < SMOOTHING_RADIUS * SMOOTHING_RADIUS {
                                let distance = dist_sq.sqrt();
                                density +=
                                    PARTICLE_MASS * smoothing_kernel(SMOOTHING_RADIUS, distance);
                            }
                        }
                    }
                }
            }
        }
        density
    }

    pub fn draw(&self, frame: &mut [u8]) {
        // --- Draw Density Field ---
        #[cfg(target_arch = "wasm32")]
        for chunk in frame.chunks_exact_mut(4) {
            chunk.copy_from_slice(&self.background_color)
        }
        #[cfg(not(target_arch = "wasm32"))]
        frame
            .par_chunks_mut(self.width as usize * BYTES_PER_PIXEL)
            .enumerate()
            .for_each(|(y, row_slice)| {
                for x in 0..self.width as usize {
                    let sample_point = Vector2D::from_array([x as f32, y as f32]);
                    // Use the visualization density function
                    let density = self.calculate_density_for_visualisation(sample_point);

                    // Normalize density and clamp
                    let intensity = (density / MAX_EXPECTED_DENSITY).min(1.0).max(0.0);

                    let idx_in_row = x * BYTES_PER_PIXEL;

                    // Blend density color with background based on intensity
                    let bg_r = self.background_color[0] as f32;
                    let bg_g = self.background_color[1] as f32;
                    let bg_b = self.background_color[2] as f32;
                    let density_r = DENSITY_COLOR[0] as f32;
                    let density_g = DENSITY_COLOR[1] as f32;
                    let density_b = DENSITY_COLOR[2] as f32;

                    // Ensure we don't write past the end of the row slice (important for last row)
                    if idx_in_row + BYTES_PER_PIXEL <= row_slice.len() {
                        row_slice[idx_in_row] =
                            (bg_r * (1.0 - intensity) + density_r * intensity) as u8;
                        row_slice[idx_in_row + 1] =
                            (bg_g * (1.0 - intensity) + density_g * intensity) as u8;
                        row_slice[idx_in_row + 2] =
                            (bg_b * (1.0 - intensity) + density_b * intensity) as u8;
                        row_slice[idx_in_row + 3] = 255; // Alpha
                    }
                }
            });

        // --- Draw Particles on Top ---
        for i in 0..self.particle_count {
            if i < self.dot_positions.len() {
                // Extra safety check
                self.draw_dot(self.dot_positions[i], self.dot_velocities[i], frame);
            }
        }
    }

    fn draw_dot(&self, position: Vector2D, velocity: Vector2D, frame: &mut [u8]) {
        let center_x = position[0] as usize;
        let center_y = position[1] as usize;
        let radius = self.dot_radius as usize;
        let feather = 1.0; // Anti-aliasing edge width

        // Calculate bounding box with bounds checking
        let min_x = center_x.saturating_sub(radius + 1);
        let min_y = center_y.saturating_sub(radius + 1);
        let max_x = (center_x + radius + 1).min(self.width as usize);
        let max_y = (center_y + radius + 1)
            .min(self.height as usize)
            .saturating_sub(1);

        // Only check pixels in the bounding box
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f32 - position[0];
                let dy = y as f32 - position[1];
                let distance = (dx * dx + dy * dy).sqrt();

                if distance <= self.dot_radius + feather {
                    let idx = (y * self.width as usize + x) * 4;
                    if idx + 3 < frame.len() {
                        let alpha = if distance <= self.dot_radius - feather {
                            1.0 // Inside circle
                        } else {
                            // Edge transition (anti-aliasing)
                            (self.dot_radius + feather - distance) / (2.0 * feather)
                        };

                        // Blend with existing color in frame buffer
                        let curr_r = frame[idx]; // Current pixel R
                        let curr_g = frame[idx + 1]; // Current pixel G
                        let curr_b = frame[idx + 2]; // Current pixel B

                        let vel = velocity;
                        let vel_mag = f32::min(
                            1.,
                            f32::abs(f32::sqrt(vel[0] * vel[0] + vel[1] * vel[1]) / 1000.),
                        );
                        let dot_r = (vel_mag * 255.) as u8; // Dot color R
                        let dot_g = 170 - (170 as f32 * vel_mag) as u8; // Dot color G
                        let dot_b = 255 - (vel_mag * 200.) as u8; // Dot color B

                        // Blend colors based on alpha
                        frame[idx] = (curr_r as f32 * (1.0 - alpha) + dot_r as f32 * alpha) as u8;
                        frame[idx + 1] =
                            (curr_g as f32 * (1.0 - alpha) + dot_g as f32 * alpha) as u8;
                        frame[idx + 2] =
                            (curr_b as f32 * (1.0 - alpha) + dot_b as f32 * alpha) as u8;
                        frame[idx + 3] = 255; // Keep alpha at 255 for the frame buffer
                    }
                }
            }
        }
    }
}

// Helper trait/impl for length_squared if Vector2D doesn't have it
trait Vec2Ext {
    fn length_squared(&self) -> f32;
}

impl Vec2Ext for Vector2D {
    #[inline]
    fn length_squared(&self) -> f32 {
        self[0] * self[0] + self[1] * self[1]
    }
}

// Equation of State (Tait equation variation)
fn convert_density_to_pressure(density: f32) -> f32 {
    let density_ratio = density / TARGET_DENSITY;
    // Clamp density ratio to avoid issues with very low densities if needed
    // let clamped_ratio = density_ratio.max(1.0); // Or some small value > 0

    let pressure = PRESSURE_MULTIPLIER * (f32::powf(density_ratio, EOS_GAMMA) - 1.0);
    pressure.max(0.1)
}

// Function to calculate interaction forces (Pressure + Viscosity) for a particle
fn calculate_interaction_forces(
    particle_index: usize,
    positions: &[Vector2D],
    velocities: &[Vector2D],
    densities: &[f32],
    grid_cells: &[Vec<usize>],
    grid_width: usize,
    grid_height: usize,
) -> Vector2D {
    let mut pressure_force = Vector2D::default();
    let mut viscosity_force = Vector2D::default(); // Initialize viscosity force

    let pos_i = positions[particle_index];
    let vel_i = velocities[particle_index]; // Get velocity_i
    let density_i = densities[particle_index];
    // Avoid division by zero if density is somehow zero
    if density_i < 1e-6 {
        return Vector2D::default();
    }

    let pressure_i = convert_density_to_pressure(density_i);
    // Precompute particle i's contribution to the shared pressure term
    let pressure_term_i = pressure_i / (density_i * density_i);

    let (scx, scy) = (
        (pos_i[0] / GRID_CELL_SIZE) as isize,
        (pos_i[1] / GRID_CELL_SIZE) as isize,
    );

    // Iterate over the 3x3 grid neighborhood
    for ny in -1..=1 {
        for nx in -1..=1 {
            let cx = scx + nx;
            let cy = scy + ny;

            // Get cell index safely
            let cell_index_opt =
                if cx >= 0 && cx < grid_width as isize && cy >= 0 && cy < grid_height as isize {
                    Some((cy * grid_width as isize + cx) as usize)
                } else {
                    None
                };

            if let Some(cell_index) = cell_index_opt {
                for neighbor_index in &grid_cells[cell_index] {
                    if *neighbor_index == particle_index {
                        continue;
                    } // Don't interact with self

                    let pos_j = positions[*neighbor_index];
                    let vel_j = velocities[*neighbor_index]; // Get velocity_j
                    let density_j = densities[*neighbor_index];
                    if density_j < 1e-6 {
                        continue;
                    } // Skip if neighbor density is zero

                    let diff = pos_i - pos_j;
                    let dist_sq = diff.length_squared();

                    // Check distance against squared radius first
                    if dist_sq < SMOOTHING_RADIUS * SMOOTHING_RADIUS && dist_sq > 1e-12 {
                        let distance = dist_sq.sqrt();
                        let direction = diff / Vector2D::splat(distance); // Normalized direction from j to i

                        // --- Pressure Force ---
                        let pressure_j = convert_density_to_pressure(density_j);
                        let pressure_term_j = pressure_j / (density_j * density_j);
                        let shared_pressure = pressure_term_i + pressure_term_j; // Use precomputed term for i

                        let slope = smoothing_kernel_derivative(SMOOTHING_RADIUS, distance);
                        pressure_force -=
                            Vector2D::splat(PARTICLE_MASS * shared_pressure * slope) * direction;

                        // --- Viscosity Force (Müller et al. 2003 style) ---
                        // Acts as a drag force based on relative velocity, weighted by the kernel value.
                        let vel_diff = vel_j - vel_i; // Difference from neighbor j to i
                        let kernel_val = smoothing_kernel(SMOOTHING_RADIUS, distance);
                        viscosity_force += Vector2D::splat(
                            VISCOSITY_STRENGTH * PARTICLE_MASS * kernel_val / density_j,
                        ) * vel_diff;
                    }
                }
            }
        }
    }

    // Final force is pressure + viscosity
    // The division by density (F/rho = a) happens in the update loop
    pressure_force + viscosity_force
}
