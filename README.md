# Toxic comments challenge resolution

## Requeriments

* conda
* tar/gzip

## Setup

**Step 1:** Create project environment.

```bash
conda env create --file environment.yml
```

**Step 2:** Extract dataset.

```bash
7z x  dataset.7z.001
```

**Step 3:** Download word embedding

```bash
mkdir word-embedding
cd word-embedding
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip *
```