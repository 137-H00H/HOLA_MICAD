# Presentation image sources

## Original README schematics and third-party credits

`aorta-schematic.svg` and `segmentation-schematic.svg` are original vector drawings created for the root README. They use simplified geometry, contain no patient data or borrowed image pixels, and illustrate concepts rather than model outputs. They are retained as earlier schematic alternatives; the root README now uses the author-supplied crimson aorta illustration.

The project author identifies the rotating anatomy in `aorta-anatomy.gif` (slide 2, `image2.gif`) as originating from **Imran et al., CIS-UNet (2024)**: [paper](https://doi.org/10.1016/j.compmedimag.2024.102470), [source repository](https://github.com/mirthAI/CIS-UNet). The same anatomy appears in `segmentation-explained.gif` and `segmentation-explained-still.png`. These legacy files and the embedded presentation copy remain; they are not displayed in the root README. Figure-specific reuse permission has not been verified. The upstream repository has an [MIT licence](https://github.com/mirthAI/CIS-UNet/blob/main/LICENSE), but its applicability to the supplied figure has not been established.

The mappings below describe extraction provenance; inclusion in the presentation does not establish authorship or a reuse licence for an image.

## Extracted presentation assets

These images were extracted from `ppt/media/` in [HOLA_Viva_Presentation.pptx](../presentations/HOLA_Viva_Presentation.pptx). Original raster files are copied without changing pixels or resolution.

| Local asset | Embedded source | Slide | Content |
| --- | --- | --- | --- |
| `axial-segmentation-examples.png` | `image4.png` | 2 | Axial CT slices with aortic overlays. |
| `positive-negative-clicks.png` | `image5.png` | 3 | Corrective click examples. |
| `cta-input.png` | `image6.png` | 4 | Input CT slice. |
| `quality-input-crop.png` | `image7.png` | 4 | CT and mask region for quality assessment. |
| `segmentation-overlay.png` | `image8.png` | 4 | CT slice with segmentation overlay. |
| `aorta-before-preprocessing.png` | `image9.png` | 5 | Vascular rendering before preprocessing. |
| `aorta-after-preprocessing.png` | `image10.png` | 5 | Aortic rendering after preprocessing. |
| `interaction-robustness.png` | `image18.png` | 6 | Click correctness and spatial-offset plots. |
| `prototype-quality-feedback.png` | `image19.png` | 7 | Quality feedback and axial/sagittal views. |
| `prototype-3d-view.png` | `image20.png` | 7 | Coronal and 3D views with click controls. |
| `segmentation-comparison.png` | `image26.png` | 9 | Six-cohort comparison across click counts and 3D predictions. |

## Derived diagrams

- `hola-workflow.svg` is a new vector diagram summarizing slide 4. It embeds the original CT, overlay, and quality-crop PNGs, with explanatory labels and a corrective feedback arrow. It illustrates the framework, not a new patient result or an executed pipeline.
- `segmentation-comparison-light.svg` places the original transparent PNG on a white SVG background so its black labels remain readable in dark mode. The underlying image is unchanged.

Both SVGs embed their raster images and have no external image dependencies. The segmentation figure retains its original 945 x 1043 resolution; enlarging it cannot add detail absent from the presentation.

The root README now focuses on the project and development process. Previous result figures remain as source assets but are not displayed in its walkthrough. The prototype screenshots and complete stopping-policy workflow describe the presentation; their interface implementation is not present in this checkout.

## Animated project walkthrough

The animations explain the project using labels checked against the current scripts. The legacy introduction and the preprocessing comparison retain presentation images; the later workflow diagrams use text and icons. They do not show a live model run, measured training progression, or newly generated patient results. They are looping GIFs. The root README uses the supplied crimson illustration for the introduction and anatomy. Workflow animations remain displayed with their still versions; preprocessing is displayed as a static comparison.

| Asset | Source and purpose |
| --- | --- |
| `aorta-anatomy.gif` | Original `ppt/media/image2.gif` from slide 2, extracted unchanged. It contains 192 frames showing rotating aortic anatomy. |
| `segmentation-explained.gif` | CT input and overlay from slide 4, with frames from the slide 2 anatomy animation. The three panels are an explanation; the anatomy is a separate example from the displayed CT. |
| `hola-feedback.gif` | Progressive reveal of CT input, mask prediction, quality estimation, human review, and separate correction/finish branches. Earlier steps remain visible. |
| `development-segmentation.gif` | Six-step development flow checked against `SplitPatients.py`, `Train.py`, `aorta_data.py`, and `Clicksim.py`. |
| `development-quality.gif` | Quality-network workflow checked against `Completeness.py` and `Completeness_classifier.py`, followed by the prototype interaction concept. |
| `preprocessing-comparison.gif` | Earlier animated comparison retained as a legacy asset; no longer displayed in the README. |

The animated diagrams were rendered with System.Drawing. Workflow cards and connectors appear progressively, with a four-second pause on the complete flow before restarting. The linked still images show the complete diagrams. Source photographs keep their original geometry. The existing before/after images cannot support a faithful new 3D rotation without the corresponding volume, mesh, or recorded turntable footage.

Public dataset source links are listed next to the corresponding rows in the root README. The local `base` cohort's public/private status is not established by the code or slides and is marked unconfirmed.

## Current supplied-illustration visuals

- `aorta-crimson.png` is an unchanged copy of the project author's supplied `Realistic crimson aorta with branching vessels.png`.
- Legacy `aorta-crimson-motion.gif` (no longer displayed) applies a gentle, looping in-plane rotation of up to three degrees to that illustration, retaining its dark background. It is a 2D presentation animation, not a generated view of hidden anatomy. `aorta-crimson-still.png` is its static counterpart.
- `segmentation-intro.gif` combines the original CT and overlay images with the static transparent aorta cutout on a white canvas. The aorta does not rotate; only the step indicators animate. The duplicate section heading and explanatory footer have been removed; only the three step labels remain. `segmentation-intro-still.png` is its static counterpart.
- `preprocessing-comparison-still.png` shows the original before/after renderings together on a white canvas, with no transition, sequence caption, or footer. The root README displays this PNG directly.

These new assets contain no frames from the legacy CIS-UNet anatomy animation. The existing figure credits above continue to identify the earlier assets.

- `aorta-transparent.png` is the static cutout made with the built-in imagegen tool. Section 2 displays this transparent PNG at a compact width of 240 pixels. `aorta-white.png` is the earlier white-backed version, no longer displayed.
- Background removal prompt: Remove only the black background and red background glow to create an actual transparent alpha PNG cutout. Preserve the aorta, thin branching vessels, orientation, shape, coloring, texture, highlights and composition. No rotation, labels, new objects, shadow or glow outside the vessel.
