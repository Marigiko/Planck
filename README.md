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