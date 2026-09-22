//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Portions copyright (c) 2020 Florian Schaefer. Licensed under the MIT License; see THIRD_PARTY_NOTICES.md.                //
// The code is an adaptation of [1] and based on the work of Schäfer et al.                                                 //
// [2], and Schäfer et al. [3].                                                                                             //
//                                                                                                                          //
// [1] https://github.com/f-t-s/cholesky_by_KL_minimization/blob/f9a7d10932c422bde9f1fcfc950321c8c7b460a2/src/SortSparse.jl //                                                                        //
// [2] Schäfer, F., Katzfuss, M. and Owhadi, H. Sparse Cholesky                                                             //
//     Factorization by Kullback--Leibler Minimization. SIAM Journal on                                                     //
//     Scientific Computing, 43(3), 2021. https://doi.org/10.1137/20M1336254                                                //
// [3] Schäfer, F., Sullivan, T.J. and Owhadi, H. Compression, Inversion                                                    //
//     and Approximate PCA of Dense Kernel Matrices at Near-Linear                                                          //
//     Computational Complexity. Multiscale Modeling & Simulation, 19(12),                                                  //
//     2021. https://doi.org/10.1137/19M129526X                                                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>


using namespace std;
namespace py = pybind11;

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// From sortSparse.h & sortSparse.c

typedef struct heapNode {
  /*distance squared to closest point that is already taken out, negative for points that are taken out*/
  double dist;
  struct heapNode** handleHandle;
  struct heapNode* leftChild;
  struct heapNode* rightChild;
  /*might not be needed:*/
  unsigned int Id;
} heapNode;

heapNode* _moveDown(heapNode* const a) {
  /*If the nodes has no children:*/
  if (a->leftChild == NULL) {
    return NULL;
  }
  /*If the node has only one child:*/
  if (a->rightChild == NULL) {
    if (a->dist < a->leftChild->dist) {
      /*swaps a with its left child*/
      const double tempDist = a->dist;
      a->dist = a->leftChild->dist;
      a->leftChild->dist = tempDist;
      *(a->handleHandle) = a->leftChild;
      *(a->leftChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->leftChild->handleHandle;
      a->leftChild->handleHandle = tempHandleHandle;

      const int tempId = a->Id;
      a->Id = a->leftChild->Id;
      a->leftChild->Id = tempId;

      return a->leftChild;
    }
    else return NULL;
  }
  /*If the node has two children:*/
  if (a->leftChild->dist > a->rightChild->dist) {
    if (a->dist < a->leftChild->dist) {
      /*swaps a with its left child*/
      const double tempDist = a->dist;
      a->dist = a->leftChild->dist;
      a->leftChild->dist = tempDist;
      *(a->handleHandle) = a->leftChild;
      *(a->leftChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->leftChild->handleHandle;
      a->leftChild->handleHandle = tempHandleHandle;

      const int tempId = a->Id;
      a->Id = a->leftChild->Id;
      a->leftChild->Id = tempId;

      return a->leftChild;
    }
    else return NULL;
  }
  else {
    if (a->dist < a->rightChild->dist) {
      /*swaps a with its right child*/
      const double tempDist = a->dist;
      a->dist = a->rightChild->dist;
      a->rightChild->dist = tempDist;
      *(a->handleHandle) = a->rightChild;
      *(a->rightChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->rightChild->handleHandle;
      a->rightChild->handleHandle = tempHandleHandle;

      const int tempId = a->Id;
      //printf( "newId = %d ", a->rightChild->Id);
      a->Id = a->rightChild->Id;
      a->rightChild->Id = tempId;


      return a->rightChild;
    }
    else return NULL;
  }
}

/*Only works as expected, if the new distance is smaller than the old one*/
void update(heapNode* target, const double newDist) {
  target->dist = newDist;
  while (target != NULL) {
    target = _moveDown(target);
  }
}

/*Initialises array of nodes with proper children and writes it into nodes*/
void heapInit(const unsigned int N, heapNode* const nodes, heapNode** const handles) {
  for (unsigned int k = 0; k < N; ++k) {
    if (2 * k + 1 >= N) {
      heapNode newnode = { INFINITY, &handles[k], NULL, NULL };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    else if (2 * k + 2 >= N) {
      heapNode newnode = { INFINITY, &handles[k], &nodes[2 * k + 1], NULL };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    else {
      heapNode newnode = { INFINITY, &handles[k], &nodes[2 * k + 1], &nodes[2 * k + 2] };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    handles[k] = &nodes[k];
    nodes[k].Id = k;
  }
}

typedef struct ijlookup {
  unsigned int pres_i;
  unsigned int N;
  unsigned int S;
  unsigned int S_Buffer;
  std::vector<unsigned int> i;
  std::vector<unsigned int> j;
} ijlookup;

void ijlookup_init(ijlookup* lookup, unsigned int N) {
  lookup->pres_i = 0;
  lookup->N = N;
  lookup->S = 0;
  lookup->S_Buffer = N;
  // std::vector owns these work buffers and raises std::bad_alloc instead of
  // allowing a null allocation to be dereferenced by the native extension.
  lookup->i.assign(static_cast<size_t>(N) + 1, 0);
  lookup->j.resize(N);
  lookup->i[0] = 0;
  lookup->i[1] = 0;
}

void ijlookup_newparent(ijlookup* const lookup) {
  ++lookup->pres_i;
  lookup->i[lookup->pres_i + 1] = lookup->i[lookup->pres_i];
}

void ijlookup_newson(ijlookup* const lookup, const unsigned int Id) {
  if (lookup->S == std::numeric_limits<unsigned int>::max()) {
    throw std::length_error("max-min child lookup exceeds native index capacity");
  }
  ++lookup->S;
  if (lookup->S > lookup->S_Buffer) {
    if (lookup->S_Buffer > std::numeric_limits<unsigned int>::max() / 2) {
      throw std::length_error("max-min child lookup exceeds native index capacity");
    }
    const unsigned int new_buffer = lookup->S_Buffer * 2;
    lookup->j.resize(new_buffer);
    lookup->S_Buffer = new_buffer;
  }
  lookup->j[lookup->S - 1] = Id;
  ++lookup->i[lookup->pres_i + 1];
}

void ijlookup_destruct(ijlookup* lookup) {
  lookup->i.clear();
  lookup->j.clear();
}

double dist(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (int k = 0; k < (int)d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return sqrt(ret);
}

double dist_2d(const unsigned int i, const unsigned int j, const double* const coords) {
  return sqrt((coords[2 * i] - coords[2 * j]) * (coords[2 * i] - coords[2 * j])
                + (coords[2 * i + 1] - coords[2 * j + 1]) * (coords[2 * i + 1] - coords[2 * j + 1]));
}

double dist_3d(const unsigned int i, const unsigned int j, const double* const coords) {
  return sqrt((coords[3 * i] - coords[3 * j]) * (coords[3 * i] - coords[3 * j])
                + (coords[3 * i + 1] - coords[3 * j + 1]) * (coords[3 * i + 1] - coords[3 * j + 1])
                + (coords[3 * i + 2] - coords[3 * j + 2]) * (coords[3 * i + 2] - coords[3 * j + 2]));
}

double dist2(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (int k = 0; k < (int)d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return ret;
}

double dist2_2d(const unsigned int i, const unsigned int j, const double* const coords) {
  return (coords[2 * i] - coords[2 * j]) * (coords[2 * i] - coords[2 * j])
  + (coords[2 * i + 1] - coords[2 * j + 1]) * (coords[2 * i + 1] - coords[2 * j + 1]);
}

double dist2_3d(const unsigned int i, const unsigned int j, const double* const coords) {
  return (coords[3 * i] - coords[3 * j]) * (coords[3 * i] - coords[3 * j])
  + (coords[3 * i + 1] - coords[3 * j + 1]) * (coords[3 * i + 1] - coords[3 * j + 1])
  + (coords[3 * i + 2] - coords[3 * j + 2]) * (coords[3 * i + 2] - coords[3 * j + 2]);
}

//void dist2Blocked(double* results, const unsigned int i, const unsigned int* const j0, const unsigned int n, const double* const coords, const unsigned int d);
//void dist2BlockedVec(double* results, const double* const ivec, const unsigned int* const j0, const unsigned int n, const double* const coords, const unsigned int d);
double dist2fix(const double* x, const unsigned int j, const double* const coords, const unsigned int d);

inline double in_dist2(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (unsigned int k = 0; k < d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return ret;
}

void  determineChildren(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int d, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);


  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2(Id, lookup->j[k], coords, d);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}

void determineChildren_2d(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);


  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2_2d(Id, lookup->j[k], coords);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}

void determineChildren_3d(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);


  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2_3d(Id, lookup->j[k], coords);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}


//void  determineChildrenBlocked(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int d, const unsigned int N, const unsigned int Id, const unsigned int iter);

void create_ordering(unsigned int* P, unsigned int* revP, double* distances, const unsigned int d, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
  * Inputs:
  *  P:
  *    An N element array containing the hierarchical ordering
  *  revP:
  *    An N element array containing the inverse of the hierarchical ordering
  *  distances:
  *    An N element array containing the distance ( length scale ) of each dof
  *  d:
  *    The number of spatial dimensions
  *  N:
  *    The number of points
  *  coords:
  *    An d*N element array that contains the different points coordinates, with the
  *    coordinates of a given point in contiguous memory
  */

  /*Allocate the heap structure with exception-safe ownership:*/
  std::vector<heapNode> nodes(N);
  std::vector<heapNode*> handles(N);
  /*Initiate the heap structure*/
  heapInit(N, nodes.data(), handles.data());
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
  *the i-th entry of parents will contain the number of its parent in the ordering */
  std::vector<unsigned int> parents(N);

  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist(rootId, k, coords, d) > distances[0]) {
      distances[0] = dist(rootId, k, coords, d);
    }
    update(handles[k], dist(rootId, k, coords, d));
      parents[k] = 0;
  }

  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles.data();
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren(
        nodes.data(), handles.data(), &lookup, parents.data(), coords, d, N, pivotId, k);
  }

  ijlookup_destruct(&lookup);
}

void create_ordering_2d(unsigned int* P, unsigned int* revP, double* distances, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
  * Inputs:
  *  P:
  *    An N element array containing the hierarchical ordering
  *  revP:
  *    An N element array containing the inverse of the hierarchical ordering
  *  distances:
  *    An N element array containing the distance ( length scale ) of each dof
  *  N:
  *    The number of points
  *  coords:
  *    An d*N element array that contains the different points coordinates, with the
  *    coordinates of a given point in contiguous memory
  */

  /*Allocate the heap structure with exception-safe ownership:*/
  std::vector<heapNode> nodes(N);
  std::vector<heapNode*> handles(N);
  /*Initiate the heap structure*/
  heapInit(N, nodes.data(), handles.data());
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
  *the i-th entry of parents will contain the number of its parent in the ordering */
  std::vector<unsigned int> parents(N);

  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist_2d(rootId, k, coords) > distances[0]) {
      distances[0] = dist_2d(rootId, k, coords);
    }
    update(handles[k], dist_2d(rootId, k, coords));
    parents[k] = 0;
  }

  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles.data();
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren_2d(
        nodes.data(), handles.data(), &lookup, parents.data(), coords, N, pivotId, k);
  }

  ijlookup_destruct(&lookup);
}


void create_ordering_3d(unsigned int* P, unsigned int* revP, double* distances, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
  * Inputs:
  *  P:
  *    An N element array containing the hierarchical ordering
  *  revP:
  *    An N element array containing the inverse of the hierarchical ordering
  *  distances:
  *    An N element array containing the distance ( length scale ) of each dof
  *  N:
  *    The number of points
  *  coords:
  *    An d*N element array that contains the different points coordinates, with the
  *    coordinates of a given point in contiguous memory
  */

  /*Allocate the heap structure with exception-safe ownership:*/
  std::vector<heapNode> nodes(N);
  std::vector<heapNode*> handles(N);
  /*Initiate the heap structure*/
  heapInit(N, nodes.data(), handles.data());
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
  *the i-th entry of parents will contain the number of its parent in the ordering */
  std::vector<unsigned int> parents(N);

  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist_3d(rootId, k, coords) > distances[0]) {
      distances[0] = dist_3d(rootId, k, coords);
    }
    update(handles[k], dist_3d(rootId, k, coords));
    parents[k] = 0;
  }

  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles.data();
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren_3d(
        nodes.data(), handles.data(), &lookup, parents.data(), coords, N, pivotId, k);
  }

  ijlookup_destruct(&lookup);
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------

// New function: MaxMincpp (integrate all c functions)
// Input: a location matrix (nrow = sample size; ncol = dimension)
// Output: the max-min order with zero-based indices.

vector<int> maxmin_order(
    py::array_t<double, py::array::c_style | py::array::forcecast> locations)
{
  py::buffer_info info = locations.request();
  if (info.ndim != 2) {
    throw py::value_error("locations must be a two-dimensional array");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw py::value_error("locations must have at least one row and one column");
  }
  if (static_cast<unsigned long long>(info.shape[0]) >
          static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
      static_cast<unsigned long long>(info.shape[1]) >
          std::numeric_limits<unsigned int>::max()) {
    throw py::value_error("locations is too large for the native ordering backend");
  }

  const unsigned int N = static_cast<unsigned int>(info.shape[0]);
  const unsigned int dim = static_cast<unsigned int>(info.shape[1]);
  const auto* locations_raw_ptr = static_cast<const double*>(info.ptr);

  // create storage for result
  auto res = std::vector<int>();
  res.reserve(N);


  // RAII-owned work arrays keep allocation failures exception-safe.
  std::vector<unsigned int> P(N);
  std::vector<unsigned int> revP(N);
  std::vector<double> distances(N);
  std::vector<double> coords(static_cast<size_t>(dim) * N);

  // Find the average point.
  unsigned int first_node;

  double cur_dist2, min_dist2;
  std::vector<double> average_arr(dim);
  for(int j = 0; j < dim; j++)
  {
    average_arr[j] = 0.0;
  }
  for (int i = 0; i < (int)N; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      const double value = locations_raw_ptr[dim * i + j];
      if (!std::isfinite(value)) {
        throw py::value_error("locations must contain only finite values");
      }
      coords[dim * i + j] = value;
      average_arr[j] += coords[dim * i + j];
    }
  }
  for(int j = 0; j < dim; j++)
  {
    average_arr[j] /= N;
  }
  min_dist2 = -1;
  first_node = -1;
  for (int i = 0; i < (int)N; i++)
  {
    cur_dist2 = 0;
    for(int j = 0; j < dim; j++)
    {
      cur_dist2 += (coords[dim * i + j] - average_arr[j]) * (coords[dim * i + j] - average_arr[j]);
    }
    if (min_dist2 < 0 || cur_dist2 < min_dist2)
    {
      min_dist2 = cur_dist2;
      first_node = i;
    }
  }

  {
    py::gil_scoped_release release;
    if (dim == 2)
    {
      create_ordering_2d(
          P.data(), revP.data(), distances.data(), N, coords.data(), first_node);
    }
    else if (dim == 3)
    {
      create_ordering_3d(
          P.data(), revP.data(), distances.data(), N, coords.data(), first_node);
    }
    else
    {
      create_ordering(
          P.data(), revP.data(), distances.data(), dim, N, coords.data(), first_node);
    }
  }

  res.push_back(first_node);
  for (int i = 1; i < N; i++) {
    res.push_back(P[i]);
  }
  return res;
}

PYBIND11_MODULE(_maxmin, m) {
  m.doc() = "Private native backend for exact max-min ordering.";
  m.def(
      "maxmin_order",
      &maxmin_order,
      py::arg("locations"),
      "Compute a zero-based max-min ordering.");
}
