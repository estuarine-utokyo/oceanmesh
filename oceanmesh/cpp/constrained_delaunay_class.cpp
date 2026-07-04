// Constrained Delaunay triangulation binding (egfix support).
//
// Mirrors delaunay_class.cpp's DelaunayTriangulation interface so
// mesh_generator can swap it in when fixed EDGES (egfix) are present
// — the OceanMesh2D high-fidelity capability that MATLAB gets from
// its built-in constrained delaunayTriangulation. Constrained edges
// are guaranteed to appear as triangulation edges, so the mesh
// boundary can be made to lie EXACTLY on a shoreline polyline chain.

#if _MSC_VER
#pragma warning(disable : 4267)
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <vector>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>

namespace py = pybind11;
using py::ssize_t;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vb = CGAL::Triangulation_vertex_base_with_info_2<unsigned int, K>;
using Fbb = CGAL::Triangulation_face_base_with_info_2<unsigned int, K>;
using Fb = CGAL::Constrained_triangulation_face_base_2<K, Fbb>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using Itag = CGAL::Exact_predicates_tag;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, Tds, Itag>;

using Point = K::Point_2;
using Cvi = CDT::Finite_vertices_iterator;

PYBIND11_MODULE(_constrained_delaunay_class, m) {
  py::class_<CDT>(m, "ConstrainedDelaunayTriangulation")

      .def(py::init())

      .def("insert",
           [](CDT &cdt, const std::vector<double> &p) {
             std::vector<std::pair<Point, unsigned>> points;
             std::size_t num_points = p.size() / 2;
             unsigned start = (unsigned)cdt.number_of_vertices();
             for (std::size_t i = 0; i < num_points; ++i) {
               points.push_back(std::make_pair(
                   Point(p[i * 2 + 0], p[i * 2 + 1]), start));
               start += 1;
             }
             cdt.insert(points.begin(), points.end());
             return cdt.number_of_vertices();
           })

      .def("insert_constraints",
           // flat [x1a, y1a, x1b, y1b,  x2a, y2a, x2b, y2b, ...]
           [](CDT &cdt, const std::vector<double> &segs) {
             std::size_t num_segs = segs.size() / 4;
             for (std::size_t i = 0; i < num_segs; ++i) {
               cdt.insert_constraint(
                   Point(segs[i * 4 + 0], segs[i * 4 + 1]),
                   Point(segs[i * 4 + 2], segs[i * 4 + 3]));
             }
             return cdt.number_of_vertices();
           })

      .def("number_of_vertices", &CDT::number_of_vertices)

      .def("number_of_faces",
           [](CDT &cdt) {
             int count = 0;
             for (CDT::Finite_faces_iterator fit =
                      cdt.finite_faces_begin();
                  fit != cdt.finite_faces_end(); ++fit)
               count += 1;
             return count;
           })

      .def("get_finite_vertices",
           [](CDT &cdt) {
             std::vector<double> verts;
             verts.resize(cdt.number_of_vertices() * 2);
             unsigned i = 0;
             for (Cvi vi = cdt.finite_vertices_begin();
                  vi != cdt.finite_vertices_end(); ++vi) {
               // renumber sequentially so the face table is dense
               vi->info() = i;
               verts[i * 2] = vi->point().x();
               verts[i * 2 + 1] = vi->point().y();
               i += 1;
             }
             ssize_t sdble = sizeof(double);
             ssize_t num_verts = verts.size() / 2;
             std::vector<ssize_t> shape = {num_verts, 2};
             std::vector<ssize_t> strides = {sdble * 2, sdble};
             return py::array(py::buffer_info(
                 verts.data(), sizeof(double),
                 py::format_descriptor<double>::format(), 2, shape,
                 strides));
           })

      .def("get_finite_cells",
           [](CDT &cdt) {
             // CALL get_finite_vertices FIRST (it renumbers infos).
             std::vector<int> faces;
             int nf = 0;
             for (CDT::Finite_faces_iterator fit =
                      cdt.finite_faces_begin();
                  fit != cdt.finite_faces_end(); ++fit)
               nf += 1;
             faces.resize(nf * 3);
             int i = 0;
             for (CDT::Finite_faces_iterator fit =
                      cdt.finite_faces_begin();
                  fit != cdt.finite_faces_end(); ++fit) {
               CDT::Face_handle face = fit;
               face->info() = i;
               faces[i * 3] = face->vertex(0)->info();
               faces[i * 3 + 1] = face->vertex(1)->info();
               faces[i * 3 + 2] = face->vertex(2)->info();
               i += 1;
             }
             ssize_t soint = sizeof(int);
             ssize_t num_faces = faces.size() / 3;
             std::vector<ssize_t> shape = {num_faces, 3};
             std::vector<ssize_t> strides = {soint * 3, soint};
             return py::array(py::buffer_info(
                 faces.data(), sizeof(int),
                 py::format_descriptor<int>::format(), 2, shape,
                 strides));
           })

      .def("get_constrained_edges",
           // vertex-info pairs of every constrained edge (after
           // get_finite_vertices renumbering)
           [](CDT &cdt) {
             std::vector<int> edges;
             for (CDT::Finite_edges_iterator eit =
                      cdt.finite_edges_begin();
                  eit != cdt.finite_edges_end(); ++eit) {
               if (cdt.is_constrained(*eit)) {
                 CDT::Face_handle f = eit->first;
                 int idx = eit->second;
                 edges.push_back(f->vertex(CDT::cw(idx))->info());
                 edges.push_back(f->vertex(CDT::ccw(idx))->info());
               }
             }
             ssize_t soint = sizeof(int);
             ssize_t num_edges = (ssize_t)edges.size() / 2;
             std::vector<ssize_t> shape = {num_edges, 2};
             std::vector<ssize_t> strides = {soint * 2, soint};
             return py::array(py::buffer_info(
                 edges.data(), sizeof(int),
                 py::format_descriptor<int>::format(), 2, shape,
                 strides));
           });
}
