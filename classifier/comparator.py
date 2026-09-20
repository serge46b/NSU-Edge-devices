from pathlib import Path
import sqlite3
import numpy as np

HERE = Path(__file__).resolve().parent
DIM = 512
DB = HERE / "db" / "faces.db"


class Comparator:
    __comparison_threshold: float
    __db_conn: sqlite3.Connection
    __all_embeddings: np.ndarray
    __all_names: dict[str, tuple[int, int]]

    def __init__(self, comparison_threshold: float) -> None:
        self.__comparison_threshold = comparison_threshold
        self.__db_conn = self._load_db(DB)
        self.__all_embeddings, self.__all_names = self._load_embeddings()

    def _load_db(self, db_path: Path) -> sqlite3.Connection:
        if not db_path.is_file():
            raise FileNotFoundError(f"Comparator: missing {db_path}")
        return sqlite3.connect(db_path)

    def _load_embeddings(self) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
        rows = self.__db_conn.execute(
            "SELECT id, name, image, embedding FROM faces ORDER BY id"
        ).fetchall()
        embs = []
        names = {}
        c_name = ""
        st_i = 0
        for i, (rid, name, _, blob) in enumerate(rows):
            vec = np.frombuffer(blob, dtype=np.float32)
            if vec.size != DIM:
                raise ValueError(f"Comparator: bad blob id={rid} bytes={len(blob)}")
            embs.append(np.array(vec, dtype=np.float32))
            if c_name == "":
                c_name = name
                st_i = i
                continue
            if name != c_name:
                names[c_name] = (st_i, i - 1)
                c_name = name
                st_i = i
        names[c_name] = (st_i, len(embs) - 1)
        return np.stack(embs), names

    def find_person(self, embeddings: np.ndarray | list[np.ndarray]) -> str | None:
        query = self._stack_query_embeddings(embeddings)
        if query is None or self.__all_embeddings.shape[0] == 0:
            return None

        dists = 1.0 - (query @ self.__all_embeddings.T)

        best_name: str | None = None
        best_dist = float("inf")
        for name, (st, end) in self.__all_names.items():
            per_person = dists[:, st : end + 1]
            score = float(per_person.min(axis=1).max())
            if score < best_dist:
                best_dist = score
                best_name = name

        if best_name is None or best_dist > self.__comparison_threshold:
            return None
        return best_name

    def _stack_query_embeddings(
        self, embeddings: np.ndarray | list[np.ndarray]
    ) -> np.ndarray | None:
        if isinstance(embeddings, list):
            if not embeddings:
                return None
            rows = [np.asarray(e, dtype=np.float32).reshape(-1) for e in embeddings]
        else:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim == 1:
                rows = [arr.reshape(-1)]
            elif arr.ndim == 2:
                if arr.shape[0] == 0:
                    return None
                return arr
            else:
                raise ValueError(
                    f"Comparator: expected 1D or 2D embedding array, got ndim={arr.ndim}"
                )
        if any(r.size != DIM for r in rows):
            raise ValueError(f"Comparator: embedding size must be {DIM}")
        return np.stack(rows)
