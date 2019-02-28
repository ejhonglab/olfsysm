\name{olfsysm-package}
\alias{olfsysm-package}
\alias{olfsysm}
\docType{package}
\title{
  D. melanogaster olfactory system modeling
}
\description{
  Provide hooks to the C++ olfsysm library
}
\details{
  Whenever possible, batch (all-odor) processing functions should be preferred
  heavily over single-odor simulation functions. Batch processing functions are
  heavily parallelized, and do not require intermediate conversions between R
  matrices and C++ (Eigen) matrices.
}
\author{
Maintainer: Matthew Bauer <bauer.s.matthew@gmail.com>
}
\keyword{ package }
\examples{
  \dontrun{
     ## Optional simple examples of the most important functions
     ## These can be in \dontrun{} and \donttest{} blocks.   
  }
}
