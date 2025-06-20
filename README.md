# Planck - Next.js Implementation

Esta es una implementación web del proyecto Planck utilizando Next.js. El proyecto original se encuentra en [https://github.com/Marigiko/Planck](https://github.com/Marigiko/Planck).

## Descripción

Planck es un sistema cuántico para mejorar conjuntos de datos y optimizar el tiempo de entrenamiento de redes neuronales. Esta implementación web proporciona una interfaz de usuario para interactuar con los algoritmos cuánticos del proyecto original.

## Características

- Interfaz de usuario moderna y responsive
- Carga de datos CSV
- Configuración de algoritmos cuánticos (QSVM, QPCA, QAOA)
- Visualización de resultados con gráficos estadísticos
- Benchmarks de rendimiento

## Tecnologías utilizadas

- Next.js 14
- React
- Tailwind CSS
- shadcn/ui
- Chart.js

## Instrucciones para conectar con el repositorio original

Para conectar esta implementación con el repositorio original y crear una rama "Next-Implementation", sigue estos pasos:

1. Clona el repositorio original:
\`\`\`bash
git clone https://github.com/Marigiko/Planck.git
cd Planck
\`\`\`

2. Crea una nueva rama llamada "Next-Implementation":
\`\`\`bash
git checkout -b Next-Implementation
\`\`\`

3. Copia todos los archivos de esta implementación web en la carpeta del repositorio clonado.

4. Añade los archivos al staging:
\`\`\`bash
git add .
\`\`\`

5. Realiza un commit con los cambios:
\`\`\`bash
git commit -m "Add Next.js implementation"
\`\`\`

6. Sube la rama al repositorio remoto:
\`\`\`bash
git push origin Next-Implementation
\`\`\`

## Estructura del proyecto

- `/app`: Páginas y rutas de la aplicación
- `/components`: Componentes reutilizables
- `/public`: Archivos estáticos

## Desarrollo

1. Instala las dependencias:
\`\`\`bash
npm install
\`\`\`

2. Inicia el servidor de desarrollo:
\`\`\`bash
npm run dev
\`\`\`

3. Abre [http://localhost:3000](http://localhost:3000) en tu navegador.

```mermaid
flowchart TD
    %% Estilos opcionales para claridad
    classDef section fill:#f1f5f9,stroke:#94a3b8,stroke-width:1px,color:#0f172a,font-size:14px;

    subgraph A[📥 Data Input]
        A1[Files: CSV, JSON, Excel]
        A2[Databases: PostgreSQL, MongoDB]
        A3[Digital Twin: BIM, IoT, GIS]
        A4[Streaming: Kafka]
    end

    subgraph B[🔁 Flow Orchestration]
        B1[Low-latency Data Feeder]
        B2[Event Triggers]
        B3[Smart Logic Routing]
        B4[Service Orchestration]
        B5[Load Balancer to AI/QC]
        B6[Validation Engine]
    end

    subgraph C[🧠 AI / Quantum Processing]
        C1[Cleaning & Normalization - pandas, Dask]
        C2[Smart Suggestion - XGBoost]
        C3[Auto Encoding - scikit-learn]
        C4[Audit - EvidentlyAI]
        C5[QasmTranspiler → OpenQASM 3.0]
        C6[Backend Routing]
    end

    subgraph D1[🧪 Simulation Backend]
        D1a[Digital Twin Simulation - SimPy]
    end

    subgraph D2[⚛️ Quantum Backend]
        D2a[OpenQASM 3.0 Execution]
        D2b[IBM Q, Amazon Braket, IonQ]
    end

    subgraph E[📊 Measurement & Analysis]
        E1[Post-processing - NumPy, Polars]
        E2[Visualization - Plotly, Streamlit]
        E3[Dispatch Scheduling]
    end

    subgraph F[🔗 API Layer]
        F1[REST APIs - FastAPI]
        F2[Web UI - React]
    end

    subgraph G[🔐 Security & Monitoring]
        G1[Auth - OAuth2, Keycloak]
        G2[Logging - ELK Stack]
        G3[Monitoring - Prometheus, Grafana]
    end

    %% Flujos
    A --> B
    B --> C
    C --> D1
    C --> D2
    D1 --> E
    D2 --> E
    E --> F
    F --> G
