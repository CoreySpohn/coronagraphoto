import math

import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation


class RenderEngine:
    def __init__(self, render_settings=None):
        # Initialize default parameters and structures
        default_render_settings = {
            "colormap": "viridis",
            "background_color": "black",
            "background_brightness": 0.0,
            "resolution": 1024,
            "figure_size": (10, 10),
            "img_dpi": 300,
            "framerate": 5,
            "animation_duration": 10,
            "output_format": "mp4",
            "output_dir": "./renders/",
            "codec": "h264",
            "extra_args": ["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
        }
        self.render_settings = default_render_settings
        if render_settings is not None:
            self.render_settings.update(render_settings)
        self.writer = FFMpegWriter(
            fps=self.render_settings["framerate"],
            codec=self.render_settings["codec"],
            extra_args=self.render_settings["extra_args"],
        )
        self.save_settings = {
            "dpi": self.render_settings["img_dpi"],
            "writer": self.writer,
        }

    def get_square_dimensions(self, count):
        # Function to determine the dimensions for a square-like grid
        root = math.sqrt(count)
        rows = math.floor(root)
        cols = math.ceil(root)
        if rows * cols < count:
            rows += 1
        return rows, cols

    def plot_for_scenario(
        self, coronagraph_subfigures, system, coronagraphs, scenario, observations, time
    ):
        nrows, ncols = self.get_square_dimensions(len(observations.obs_wavelengths))
        for coronagraph_subfig, coronagraph in zip(
            coronagraph_subfigures, coronagraphs
        ):
            wavelength_subfigs = coronagraph_subfig.subfigures(nrows, ncols)
            for wavelength_subfig, wavelength in zip(
                wavelength_subfigs.flatten(), observations.obs_wavelengths
            ):
                # subscenario = scenario.get_subscenario(time, wavelength)
                ax = wavelength_subfig.subplots()
                self.plot_image(ax, observations, time, wavelength)
            # else:
            #     self.plot_image(
            #         wavelength_axes, observations.images[(time, wavelength)]
            #     )

    def render(self, system, coronagraphs, scenario, observations):
        if not isinstance(coronagraphs, list):
            coronagraphs = [coronagraphs]

        is_animation = len(observations.obs_times) > 1
        nrows, ncols = self.get_square_dimensions(len(coronagraphs))
        fig = plt.figure()
        coronagraph_subfigures = [fig.subfigures(nrows, ncols) for _ in coronagraphs]

        if is_animation:

            def update(frame_time):
                self.plot_for_scenario(
                    coronagraph_subfigures,
                    system,
                    coronagraphs,
                    scenario,
                    observations,
                    frame_time,
                )
                plt.suptitle(f"Time: {frame_time.decimalyear:.2f}")
                return [
                    ax for subfig in coronagraph_subfigures for ax in subfig.get_axes()
                ]

            anim = FuncAnimation(fig, update, frames=observations.obs_times)
            anim.save(
                f"renders/{coronagraphs[0].name}_animation.mp4", **self.save_settings
            )
            return anim
        else:
            # Static image generation
            self.plot_for_scenario(
                coronagraph_subfigures,
                system,
                coronagraphs,
                scenario,
                observations,
                observations.obs_times[0],
            )
            plt.show()
            return fig

    def plot_image(self, ax, observations, time, wavelength):
        photon_counts = observations.images[(time, wavelength)]
        # Convert photon_counts into an image and plot on 'ax'
        ax.imshow(photon_counts, cmap="viridis")
        ax.set_title(f"{wavelength.to(u.nm).value:.0f} nm")
