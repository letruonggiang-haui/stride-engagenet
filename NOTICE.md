# What this licence does and does not cover

The `LICENSE` file, version 3 of the GNU General Public License, covers the code
in this repository.

It does not cover, and this repository does not redistribute:

* **Pretrained backbone weights.** SVFAP and MAE-DFER weights come from their own
  authors under their own terms.
* **The InsightFace ONNX models.** `det_10g` for detection and `1k3d68` for the
  68-point three-dimensional landmarks come from the `buffalo_l` pack under the
  InsightFace project's terms.
* **Dataset content.** EngageNet and DAiSEE video, frames, face crops, landmark
  caches, backbone features and per-clip prediction arrays. The datasets come
  from their own authors; the README says how each is obtained.

`code/geom_extract_min.py` is a verbatim extract of the geometry module of a
sibling project by the same authors, released here under the same terms as the
rest of this repository.
