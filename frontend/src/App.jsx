
import { useState } from "react";
import "./App.css";

const API_URL = (
  import.meta.env.VITE_API_URL ||
  "https://poker-api-1-0.onrender.com"
).replace(/\/+$/, "");

const DATASETS = {
  "25": {
    label: "Dataset al 25 %",
    records: "6,253 registros",
    url: "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156193/poker-hand-training-25.csv",
  },
  "50": {
    label: "Dataset al 50 %",
    records: "12,503 registros",
    url: "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156194/poker-hand-training-50.csv",
  },
  "100": {
    label: "Dataset al 100 %",
    records: "25,010 registros",
    url: "https://res.cloudinary.com/lq5tulzu/raw/upload/v1793156194/poker-hand-training-100.csv",
  },
};

const HANDS = [
  { suit: "1", rank: "1" },
  { suit: "2", rank: "5" },
  { suit: "3", rank: "7" },
  { suit: "4", rank: "9" },
  { suit: "1", rank: "13" },
];

const SUITS = [
  { value: "1", label: "Corazones" },
  { value: "2", label: "Diamantes" },
  { value: "3", label: "Tréboles" },
  { value: "4", label: "Picas" },
];

const RANKS = Array.from({ length: 13 }, (_, index) => ({
  value: String(index + 1),
  label: String(index + 1),
}));

function App() {
  const [section, setSection] = useState("predict");
  const [cards, setCards] = useState(HANDS);
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  const [dataset, setDataset] = useState("50");
  const [tag, setTag] = useState("poker-50");
  const [retrainKey, setRetrainKey] = useState("");
  const [retraining, setRetraining] = useState(false);

  const [message, setMessage] = useState(null);

  function updateCard(index, field, value) {
    setCards((previous) =>
      previous.map((card, i) =>
        i === index ? { ...card, [field]: value } : card
      )
    );
  }

  async function predictHand(event) {
    event.preventDefault();
    setPredicting(true);
    setPrediction(null);
    setMessage(null);

    const payload = {};

    cards.forEach((card, index) => {
      payload[`suit_card_${index + 1}`] = Number(card.suit);
      payload[`rank_card_${index + 1}`] = Number(card.rank);
    });

    try {
      const response = await fetch(`${API_URL}/score`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(
          `La API respondió con HTTP ${response.status}`
        );
      }

      setPrediction(await response.json());
    } catch (error) {
      setMessage({
        type: "error",
        text: `No se pudo obtener la predicción. ${error.message}`,
      });
    } finally {
      setPredicting(false);
    }
  }

  async function retrainModel(event) {
    event.preventDefault();
    setMessage(null);

    if (!tag.trim()) {
      setMessage({
        type: "error",
        text: "Escribe una etiqueta para identificar la imagen.",
      });
      return;
    }

    if (!retrainKey.trim()) {
      setMessage({
        type: "error",
        text: "Introduce la clave de acceso al servicio.",
      });
      return;
    }

    setRetraining(true);

    try {
      const response = await fetch(`${API_URL}/retrain`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Retrain-Key": retrainKey.trim(),
        },
        body: JSON.stringify({
          dataseturl: DATASETS[dataset].url,
          sha: tag.trim(),
        }),
      });

      const result = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
          result.detail ||
          `La API respondió con HTTP ${response.status}`
        );
      }

      setMessage({
        type: "success",
        text:
          result.message ||
          "Solicitud enviada. Revisa GitHub Actions para consultar el progreso.",
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: `No se pudo iniciar el reentrenamiento. ${error.message}`,
      });
    } finally {
      setRetraining(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-symbol">♠</span>
          <span>
            <strong>Poker ModelOps</strong>
            <small>Machine Learning Lab</small>
          </span>
        </a>

        <span className="status-pill">
          <span className="status-dot" />
          XGBoost API
        </span>
      </header>

      <main className="container">
        <section className="intro">
          <p className="eyebrow">MODEL OPERATIONS / POKER</p>
          <h1>Tu modelo, tus datos, tus predicciones.</h1>
          <p>
            Prueba el clasificador de manos de póker y administra el
            reentrenamiento de tu modelo XGBoost.
          </p>
        </section>

        <section className="stats-grid">
          <article className="stat-card">
            <span className="stat-icon">♠</span>
            <div>
              <p>Modelo</p>
              <strong>XGBoost</strong>
            </div>
          </article>

          <article className="stat-card">
            <span className="stat-icon">▤</span>
            <div>
              <p>Dataset completo</p>
              <strong>25,010 registros</strong>
            </div>
          </article>

          <article className="stat-card">
            <span className="stat-icon">↻</span>
            <div>
              <p>Automatización</p>
              <strong>GitHub Actions</strong>
            </div>
          </article>
        </section>

        <nav className="tabs" aria-label="Secciones de ModelOps">
          <button
            type="button"
            className={section === "predict" ? "tab active" : "tab"}
            onClick={() => {
              setSection("predict");
              setMessage(null);
            }}
          >
            ♠ Predicción de manos
          </button>

          <button
            type="button"
            className={section === "retrain" ? "tab active" : "tab"}
            onClick={() => {
              setSection("retrain");
              setMessage(null);
            }}
          >
            ↻ Reentrenar modelo
          </button>
        </nav>

        {message && (
          <div
            className={`notice ${message.type}`}
            role="status"
            aria-live="polite"
          >
            {message.text}
          </div>
        )}

        {section === "predict" ? (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">PREDICTOR</p>
                <h2>Analizar una mano</h2>
                <p>
                  Selecciona el palo y el valor de cada una de las cinco cartas.
                </p>
              </div>

              <span className="panel-badge">POST /score</span>
            </div>

            <form onSubmit={predictHand}>
              <div className="cards-grid">
                {cards.map((card, index) => (
                  <article className="playing-card" key={index}>
                    <div className="card-number">
                      <span>CARTA</span>
                      <strong>0{index + 1}</strong>
                    </div>

                    <label>
                      Palo
                      <select
                        value={card.suit}
                        onChange={(event) =>
                          updateCard(index, "suit", event.target.value)
                        }
                      >
                        {SUITS.map((suit) => (
                          <option key={suit.value} value={suit.value}>
                            {suit.value} — {suit.label}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label>
                      Valor
                      <select
                        value={card.rank}
                        onChange={(event) =>
                          updateCard(index, "rank", event.target.value)
                        }
                      >
                        {RANKS.map((rank) => (
                          <option key={rank.value} value={rank.value}>
                            {rank.label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </article>
                ))}
              </div>

              <button
                type="submit"
                className="primary-button"
                disabled={predicting}
              >
                {predicting ? "Consultando modelo..." : "Predecir mano"}
                <span>→</span>
              </button>
            </form>

            {prediction && (
              <div className="prediction-result" role="status">
                <span className="result-icon">✓</span>
                <div>
                  <p>Resultado de la predicción</p>
                  <h3>{prediction.hand_name}</h3>
                  <small>Clase del modelo: {prediction.prediction}</small>
                </div>
              </div>
            )}
          </section>
        ) : (
          <section className="panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">PIPELINE AUTOMATIZADO</p>
                <h2>Reentrenar modelo con GitHub Actions</h2>
                <p>
                  Selecciona una versión del dataset alojado en Cloudinary.
                </p>
              </div>

              <span className="panel-badge">ModelOps</span>
            </div>

            <form onSubmit={retrainModel} className="retrain-form">
              <label>
                Versión del dataset
                <select
                  value={dataset}
                  onChange={(event) => setDataset(event.target.value)}
                >
                  {Object.entries(DATASETS).map(([key, value]) => (
                    <option key={key} value={key}>
                      {value.label} — {value.records}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                URL directa de Cloudinary
                <input
                  value={DATASETS[dataset].url}
                  readOnly
                  aria-label="URL del dataset seleccionado"
                />
              </label>

              <div className="form-columns">
                <label>
                  Etiqueta de la imagen Docker
                  <input
                    value={tag}
                    onChange={(event) => setTag(event.target.value)}
                    maxLength={128}
                    pattern="[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
                    title="Usa letras, números, guiones, puntos o guiones bajos."
                    required
                  />
                </label>

                <label>
                  Repositorio GitHub
                  <input
                    value="IanJS7/machine-learning-poker"
                    readOnly
                  />
                </label>
              </div>

              <label>
                Clave de acceso al servicio
                <input
                  type="password"
                  value={retrainKey}
                  onChange={(event) => setRetrainKey(event.target.value)}
                  autoComplete="off"
                  placeholder="Introduce la clave configurada en Render"
                  required
                />
              </label>

              <div className="security-note">
                <span>🔒</span>
                <p>
                  La clave de acceso se envía con la solicitud. El token
                  de GitHub permanece en el backend y no se incluye en React.
                </p>
              </div>

              <button
                type="submit"
                className="primary-button"
                disabled={retraining}
              >
                {retraining
                  ? "Enviando solicitud..."
                  : "Disparar reentrenamiento"}
                <span>→</span>
              </button>
            </form>

            <p className="footnote">
              El entrenamiento se ejecuta en GitHub Actions. Puedes consultar
              el estado en la pestaña Actions de tu repositorio.
            </p>
          </section>
        )}

        <footer>
          <span>Poker ModelOps · Proyecto de Machine Learning</span>
          <a
            href="https://github.com/IanJS7/machine-learning-poker/actions"
            target="_blank"
            rel="noreferrer"
          >
            Ver ejecuciones en GitHub ↗
          </a>
        </footer>
      </main>
    </div>
  );
}

export default App;
