import numpy as np
import pandas as pd
import anndata
import celloracle as co


def test_import_and_version():
    assert isinstance(co.__version__, str) and co.__version__


def test_oracle_pca_knn_imputation_shapes():
    rs = np.random.RandomState(0)
    n_cells, n_genes = 80, 60
    X = rs.poisson(1.0, size=(n_cells, n_genes)).astype(np.float32)
    adata = anndata.AnnData(X=X)
    adata.obs_names = [f"c{i}" for i in range(n_cells)]
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["cluster"] = pd.Categorical(["a"] * (n_cells // 2) + ["b"] * (n_cells // 2))
    adata.obsm["X_umap"] = rs.normal(size=(n_cells, 2))
    oracle = co.Oracle()
    oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="cluster", embedding_name="X_umap")
    oracle.perform_PCA(n_components=10)
    oracle.knn_imputation(n_pca_dims=10, k=5, balanced=True, b_sight=20, b_maxl=10, n_jobs=1)
    assert oracle.pcs.shape == (n_cells, 10)
    assert oracle.adata.layers["imputed_count"].shape == (n_cells, n_genes)
