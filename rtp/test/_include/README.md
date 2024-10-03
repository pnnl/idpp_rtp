The `test_idpp.db` file in this directory is a subset of the full IdPP database with only
the data from Huber et al datasets (former src_id 270 and 271 -> meta src_id 554). This
should be enough to test out the package and having it built into the test subpackage in
this way makes the tests fully portable, no longer relying on hard-coded paths to large
databases somewhere else in the filesystem. If/when packaging this code into a proper
package for distribution on PyPI, the package build will need to be configured to include
this directory in the built wheels because it does not contain Python code but is necessary
for running unit tests.