# Presentation image sources

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

Reported headline values in the root README come from slide 6. The prototype screenshots and complete stopping-policy workflow describe the presentation; their interface implementation is not present in this checkout.
