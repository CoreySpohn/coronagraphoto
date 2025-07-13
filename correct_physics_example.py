#!/usr/bin/env python3
"""
Example showing the correct physics implementation.

This demonstrates:
1. How to properly evaluate flux from ExoVista at specific times/wavelengths
2. How different sources interact differently with the coronagraph
3. How to use flux_conserving_affine for resampling

NOTE: This is pseudocode showing the patterns from observation.py,
      not runnable code. Focus on the physics patterns!
"""

# Key insights from observation.py:

def gen_star_count_rate_example():
    """From observation.py - shows how star interacts with coronagraph."""
    # 1. Get stellar angular diameter in lambda/D units
    stellar_diam_lod = system.star.angular_diameter.to(
        u.lod, equiv.lod(wavelength, diameter)
    )
    
    # 2. Get stellar intensity map from coronagraph
    stellar_intens = coronagraph.stellar_intens(stellar_diam_lod).T
    
    # 3. Get star flux at specific wavelength and time
    star_flux_density = system.star.spec_flux_density(wavelength, time)
    
    # 4. Convert to photon flux
    flux_term = (star_flux_density * illuminated_area * bandwidth).decompose()
    
    # 5. Apply stellar intensity map
    count_rate = np.multiply(stellar_intens, flux_term).T
    
    return count_rate


def gen_planet_count_rate_example():
    """From observation.py - shows how planets interact with coronagraph."""
    # 1. Propagate orbits to get positions
    orbit_dataset = system.propagate(time, prop="nbody", ref_frame="helio-sky")
    
    # 2. Convert to pixel coordinates
    xyplanet = get_planet_pixel_coords(orbit_dataset)
    planet_xy_separations = (xyplanet - xystar) * pixscale
    
    # 3. Get flux for each planet
    planet_count_rate = np.zeros((npixels, npixels))
    for i, (x, y) in enumerate(planet_xy_separations):
        # Get planet flux at specific wavelength and time
        planet_flux = planet.spec_flux_density(wavelength, time)
        
        # Get off-axis PSF for this position
        psf = coronagraph.offax(x, y, lam=wavelength, D=diameter)
        
        # Apply PSF to flux
        planet_count_rate += planet_flux * psf
    
    return planet_count_rate


def gen_disk_count_rate_example():
    """From observation.py - shows how disk interacts with coronagraph."""
    # 1. Get disk flux at specific wavelength and time
    disk_image = system.disk.spec_flux_density(wavelength, time)
    
    # 2. Calculate zoom factor from disk pixels to coronagraph pixels
    zoom_factor = (
        (u.pixel * system.star.pixel_scale.to(rad_per_pix_unit)).to(
            u.lod, equiv.lod(wavelength, diameter)
        ) / coronagraph.pixel_scale
    ).value
    
    # 3. Convert to photons
    disk_image_photons = (
        disk_image.to(spec_flux_density_unit) * illuminated_area * bandwidth
    ).value
    
    # 4. Resample using existing utility (not creating new system!)
    scaled_disk = util.zoom_conserve_flux(disk_image_photons, zoom_factor)
    
    # 5. Center and crop to coronagraph size
    scaled_disk = center_and_crop(scaled_disk, coronagraph.npixels)
    
    # 6. Convolve with PSF datacube
    count_rate = compute_disk_image(scaled_disk, coronagraph.psf_datacube)
    
    return count_rate


# The key pattern for our architecture:

class CorrectPhysicsPath:
    """A light path that properly handles the three source types."""
    
    def process_star(self, system, wavelength, time, coronagraph):
        """Process star through its specific coronagraph interaction."""
        # Evaluate flux NOW, not ahead of time
        star_flux = system.star.spec_flux_density(wavelength, time)
        
        # Get stellar diameter
        stellar_diam_lod = system.star.angular_diameter.to(u.lod, ...)
        
        # Apply stellar intensity map
        stellar_intens = coronagraph.stellar_intens(stellar_diam_lod)
        
        return star_flux * stellar_intens
    
    def process_planets(self, system, wavelength, time, coronagraph):
        """Process planets through their specific coronagraph interaction."""
        result = np.zeros((coronagraph.npixels, coronagraph.npixels))
        
        # Propagate orbits
        positions = system.propagate(time)
        
        for planet, (x, y) in zip(system.planets, positions):
            # Evaluate flux NOW
            planet_flux = planet.spec_flux_density(wavelength, time)
            
            # Get off-axis PSF
            psf = coronagraph.offax(x, y, lam=wavelength)
            
            result += planet_flux * psf
            
        return result
    
    def process_disk(self, system, wavelength, time, coronagraph):
        """Process disk through its specific coronagraph interaction."""
        # Evaluate flux NOW
        disk_flux_map = system.disk.spec_flux_density(wavelength, time)
        
        # Resample using EXISTING flux_conserving_affine
        from coronagraphoto.transforms.image_transforms import flux_conserving_affine
        
        resampled = flux_conserving_affine(
            disk_flux_map,
            pixscale_src=system.disk.pixel_scale,
            pixscale_tgt=coronagraph.pixel_scale,
            shape_tgt=(coronagraph.npixels, coronagraph.npixels)
        )
        
        # Convolve with PSF datacube
        result = np.einsum('ij,ijxy->xy', resampled, coronagraph.psf_datacube)
        
        return result


print("""
KEY TAKEAWAYS:
=============

1. DON'T pre-create xarray datasets - evaluate flux at execution time
2. DON'T create new resampling systems - use flux_conserving_affine
3. DO handle star/planets/disk differently in coronagraph
4. DO follow the proven patterns from observation.py

The >> pipeline pattern is great for the user interface,
but internally we need source-aware processing!
""")