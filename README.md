Explicación General
Carga de Datos y Preprocesamiento:

Se permite cargar un archivo CSV que se procesa para extraer las características (datos) y las etiquetas.
Los datos se escalan usando StandardScaler.
Interfaz de Usuario con Flet:

Se implementa una vista de login, una vista de configuración y vistas de resultados.
La vista de configuración permite seleccionar entre tres tipos de "ansatz" cuánticos:
QSVM: Utiliza un mapa de características cuántico (compuesto de puertas Hadamard y rotaciones Ry) para calcular un kernel cuántico y entrenar un clasificador SVM.
QPCA: Convierte los datos clásicos a un formato cuántico (normalización) y aplica un circuito sencillo (puertas Ry) para obtener expectativas, a las que luego se les aplica PCA para reducir dimensiones.
QAOA: Resuelve el problema de Max-Cut mediante un circuito QAOA, que alterna capas de operador de costo (basado en CNOT y Rz) y operador de mezcla (Rx), y se optimiza la función de costo con COBYLA.
Ejecución de Tareas Cuánticas en Segundo Plano:

Las tareas pesadas se ejecutan en un hilo separado, mostrando una vista de carga mientras se procesan los cálculos.
Visualización de Resultados:

Dependiendo del circuito seleccionado, se muestran gráficos generados con matplotlib que ilustran los resultados (precisión en QSVM, reducción de dimensiones en QPCA, o parámetros y energía en QAOA).

```mermaid
flowchart TD
    %% Custom styles
    classDef block fill:#ffffff,stroke:#94a3b8,stroke-width:2px,color:#0f172a,font-size:14px;
    classDef highlight fill:#d1fae5,stroke:#10b981,stroke-width:2px,color:#065f46,font-weight:bold;

    %% Blocks
    A[📥 **Data Input**<br><small>Sensors, Files, Systems</small>] --> B[🧠 **AI Processing**<br><small>Predictive Models</small>]
    B --> C[📊 **Generated Results**<br><small>Simulations, Recommendations</small>]
    C --> D[🧑‍💻 **User Interaction**<br><small>Validations, Adjustments, Feedback</small>]
    D --> E[🔁 **Feedback**<br><small>Automatically Collected</small>]
    E --> F[📈 **Model Improvement**<br><small>Continuous Training</small>]
    F --> B

    %% Informative loop
    E -.->|🧪 Continuous Learning| B

    %% Apply styles
    class A,B,C,D,E block;
    class F highlight;

