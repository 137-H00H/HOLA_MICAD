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

The root README now focuses on the project and development process. Previous result figures remain as source assets but are not displayed in its walkthrough. The prototype screenshots and complete stopping-policy workflow describe the presentation; their interface implementation is not present in this checkout.

## Animated project walkthrough

The animations explain the project using labels checked against the current scripts. The introduction and preprocessing comparison retain presentation images; the later workflow diagrams use text and icons. They do not show a live model run, measured training progression, or newly generated patient results. They are looping GIFs so the README can display motion without JavaScript. Each new diagram also has a `-still.png` version for readers who prefer a static figure.

| Asset | Source and purpose |
| --- | --- |
| `aorta-anatomy.gif` | Original `ppt/media/image2.gif` from slide 2, extracted unchanged. It contains 192 frames showing rotating aortic anatomy. |
| `segmentation-explained.gif` | CT input and overlay from slide 4, with frames from the slide 2 anatomy animation. The three panels are an explanation; the anatomy is a separate example from the displayed CT. |
| `hola-feedback.gif` | Progressive reveal of CT input, mask prediction, quality estimation, human review, and separate correction/finish branches. Earlier steps remain visible. |
| `development-segmentation.gif` | Six-step development flow checked against `SplitPatients.py`, `Train.py`, `aorta_data.py`, and `Clicksim.py`. |
| `development-quality.gif` | Quality-network workflow checked against `Completeness.py` and `Completeness_classifier.py`, followed by the prototype interaction concept. |
| `preprocessing-comparison.gif` | Original before/after images from slide 5 revealed in sequence: before image, connecting arrow, then after image. The source images have fixed viewing angles; this is not a 3D turntable render. |

The diagrams were rendered with System.Drawing. Workflow cards and connectors appear progressively, with a four-second pause on the complete flow before restarting. The linked still images show the complete diagrams. The introduction retains its existing animation; source photographs keep their original geometry. The existing before/after images cannot support a faithful new 3D rotation without the corresponding volume, mesh, or recorded turntable footage.

Public dataset source links are listed next to the corresponding rows in the root README. The local `base` cohort's public/private status is not established by the code or slides and is marked unconfirmed.