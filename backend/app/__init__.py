"""FastAPI service layer (Phase 1).

The package is split along three responsibilities:

    schemas.py  -> The API contract. Validates that input is *physically and
                   logically* well formed (Pydantic). Knows nothing about the
                   model.
    model.py    -> Artifact loading, schema normalisation, prediction, SHAP.
                   Knows nothing about HTTP.
    main.py     -> The thin HTTP layer that joins the two (endpoints + CORS).

The point of the split is to keep `model.py` testable without FastAPI, and to
make "where does validation happen?" a question with exactly one answer.
"""
