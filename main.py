import flet as ft
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import Estimator
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib
from flet.matplotlib_chart import MatplotlibChart
from qiskit.quantum_info import SparsePauliOp
import networkx as nx
from scipy.optimize import minimize
import pandas as pd
import os

matplotlib.use('svg')

def main(page: ft.Page):
    # Configuración inicial de la página
    page.title = "Planck.Quantum PoC"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    global data_scaled, labels

    # Variables iniciales
    data_scaled = None
    labels = None
    
    # Función para cargar datos desde un archivo CSV.
    def load_csv(e):
        file_picker.pick_files(allow_multiple=False)
    
    # Procesa el archivo CSV seleccionado.
    def process_csv(e):
        global data_scaled, labels
        
        if e.files:
            file_path = e.files[0].path
            df = pd.read_csv(file_path)

            try:
                # Suponiendo que el último atributo es la etiqueta
                labels = df.iloc[:, -1].values
                data = df.iloc[:, :-1].values

                # Escalando los datos
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)

                # Actualizar la gráfica
                page.views.clear()
                page.views.append(configuration_view())
                page.update()

                # Usando Page.overlay.append para mostrar el SnackBar
                snack_bar = ft.SnackBar(ft.Text("Archivo CSV cargado correctamente."))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
            except Exception as ex:
                snack_bar = ft.SnackBar(ft.Text(f"Error al procesar el CSV: {ex}"))
                page.overlay.append(snack_bar)
                snack_bar.open = True
                page.update()
        else:
            snack_bar = ft.SnackBar(ft.Text("No se seleccionó ningún archivo."))
            page.overlay.append(snack_bar)
            snack_bar.open = True
            page.update() 

    file_picker = ft.FilePicker(on_result=process_csv)
    page.overlay.append(file_picker)

    # Función de validación de login
    def validate_login(e):
        if username_field.value == "Planck" and password_field.value == "Planck123":
            page.views.clear()
            page.views.append(configuration_view())
            page.update()
        else:
            error_message.value = "Credenciales incorrectas."
            error_message.update()

    # Vista de inicio de sesión
    username_field = ft.TextField(label="Usuario o email", width=300)
    password_field = ft.TextField(label="Contraseña", password=True, width=300)
    error_message = ft.Text(value="", color="red")

    login_view = ft.Column(
        controls=[
            ft.Text("Planck.Quantum", size=30, weight=ft.FontWeight.BOLD),
            username_field,
            password_field,
            ft.ElevatedButton("Iniciar sesión", on_click=validate_login),
            error_message,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
    )
    
    def generate_plot():
        """Genera un scatter plot de los datos cargados."""
        if data_scaled is None or labels is None:
            return ft.Text("Cargue un archivo CSV para visualizar los datos.")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis')
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_title('Datos cargados del CSV')
        fig.colorbar(scatter, label='Clases')

        return MatplotlibChart(fig, expand=True)

    def generate_qsvm_plot():
        # Crear la figura y los ejes
        fig, ax = plt.subplots(figsize=(6, 4))

        # Visualización de los resultados
        scatter_real = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, label='Clases reales', cmap='viridis')
        scatter_pred = ax.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c=y_pred,
            marker='x',
            label='Predicciones QSVM',
            alpha=0.5,
            cmap='cool',
        )

        # Etiquetas y título
        ax.set_xlabel('Característica 1')
        ax.set_ylabel('Característica 2')
        ax.set_title('Clasificación QSVM (con kernel cuántico estimado)')
        ax.legend()

        # Agregar una barra de color (opcional)
        fig.colorbar(scatter_real, ax=ax, label='Clases reales')

        # Retornar la gráfica como un control MatplotlibChart de Flet
        return MatplotlibChart(fig, expand=True)

    def generate_qpca_plot():
        # Crear la figura y el eje
        fig, ax = plt.subplots(figsize=(6, 2))  # Dimensiones más alargadas para un gráfico unidimensional

        # Gráfico de dispersión unidimensional
        scatter = ax.scatter(
            reduced_data,
            np.zeros_like(reduced_data),
            c=labels,
            cmap='viridis'
        )

        # Configuración del gráfico
        ax.set_xlabel('Componente Principal 1')
        ax.set_yticks([])  # Eliminar valores en el eje Y porque no tienen significado
        ax.set_title('Datos Iris después de qPCA (1 Componente Principal)')

        # Barra de color para las clases
        fig.colorbar(scatter, ax=ax, label='Clases')

        # Retornar la figura como un control de Flet
        return MatplotlibChart(fig, expand=True)

    # Vista de configuración
    def configuration_view():
        # Crear los controles de la vista
        dropdown = ft.Dropdown(
            label="Tipo de circuito ansatz",
            options=[
                ft.dropdown.Option("QAOA"),
                ft.dropdown.Option("QSVM"),
                ft.dropdown.Option("QPCA"),
            ],
        )
        qubits_field = ft.TextField(label="Cantidad de qubits", width=300)
        repeticiones_field = ft.TextField(label="Repeticiones", width=300)
        
        def execute_quantum_task(e):
            # Obtener los valores de los campos
            circuito_ansatz = dropdown.value
            cantidad_qubits = qubits_field.value
            repeticiones = repeticiones_field.value
            
            # Llamar a la función run_quantum_pipeline con los datos obtenidos
            run_quantum_pipeline(circuito_ansatz, cantidad_qubits, repeticiones)
            
            # Cambiar la vista después de ejecutar la tarea cuántica
            page.views.clear()
            page.views.append(result_view(circuito_ansatz, cantidad_qubits, repeticiones))
            page.update()
            
        # Retornar la vista con los controles
        return ft.Column(
            controls=[
                ft.Text("Configuración", size=30, weight=ft.FontWeight.BOLD),
                dropdown,
                qubits_field,
                repeticiones_field,
                ft.ElevatedButton("Ejecutar", on_click=execute_quantum_task),
                ft.ElevatedButton("Cargar CSV", on_click=load_csv),
                generate_plot(),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    # Vista de resultados
    def result_view(circuito_ansatz, cantidad_qubits, repeticiones):
        if(circuito_ansatz == 'QSVM'):
            return ft.Column(
                controls=[
                    ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD),
                    ft.Text(f"El dataset ha sido reducido con exito utilizando {cantidad_qubits} qubits y {repeticiones} repeticiones."),
                    ft.Text("La precisión del modelo cuántico ha sido calculada."),
                    ft.Text(f"Precisión: {quantum_accuracy:.2f}%"),
                    generate_qsvm_plot(),
                    ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close()),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            
        if(circuito_ansatz == 'QPCA'):
            return ft.Column(
                controls=[
                    ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD),
                    ft.Text(f"Se redujo las dimensiones del dataset con exito utilizando {cantidad_qubits} qubits y {repeticiones} repeticiones."),
                    generate_qpca_plot(),
                    ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close()),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )
            
        if(circuito_ansatz == 'QAOA'):
            return ft.Column(
                controls=[
                    ft.Text("Resultados", size=30, weight=ft.FontWeight.BOLD),
                    ft.Text(f"La optimizacion de parametros se ejecuto exitosamente utilizando {cantidad_qubits} qubits y {repeticiones} repeticiones."),
                    ft.Text(f"Los Parámetros óptimos son: {params} y la Energía mínima encontrada es: {energy}"),
                    ft.ElevatedButton("Finalizar", on_click=lambda e: page.window_close()),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            )

    # Función principal de procesamiento cuántico
    def run_quantum_pipeline(circuito_ansatz, cantidad_qubits, repeticiones):
        global quantum_accuracy  # Para acceder desde la vista de resultados
        global X_train, X_test, y_train, y_test, y_pred
        global reduced_data
        global params, energy
        
        if(circuito_ansatz == 'QSVM'):
            
            # Dividir los datos en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)
            
            # Definir un mapa de características cuántico (QSVM)
            def feature_map(num_qubits):
                qc = QuantumCircuit(num_qubits)
                for i in range(num_qubits):
                    qc.h(i)  # Aplicar Hadamard
                    qc.ry(i, i)  # Rotaciones parametrizadas por características
                qc.barrier()
                return qc

            n_qubits = int(cantidad_qubits) #X_train.shape[1]
            quantum_feature_map = feature_map(n_qubits)

            # Kernel cuántico estimado (simulado aquí con un kernel clásico)
            def quantum_kernel_estimation(x1, x2):
                return np.dot(x1, x2.T)

            # Entrenar un modelo SVM clásico con el kernel cuántico estimado
            svc = SVC(kernel='precomputed')
            kernel_matrix_train = quantum_kernel_estimation(X_train, X_train)
            svc.fit(kernel_matrix_train, y_train)

            # Evaluar el modelo en los datos de prueba
            kernel_matrix_test = quantum_kernel_estimation(X_test, X_train)
            y_pred = svc.predict(kernel_matrix_test)

            quantum_accuracy = accuracy_score(y_test, y_pred) * 100
        
        elif(circuito_ansatz == 'QPCA'):
            
            # Configurar el simulador cuántico
            simulator = AerSimulator()

            # Definir un estimador para calcular expectativas
            estimator = Estimator()
            
            # Transformar datos clásicos a un espacio cuántico
            def classical_to_quantum(data):
                n_features = data.shape[1]
                quantum_data = []
                for row in data:
                    # Normalizar cada fila
                    normalized_row = row / np.linalg.norm(row)
                    quantum_data.append(normalized_row)
                return np.array(quantum_data)

            quantum_data = classical_to_quantum(data_scaled)

            # Definir un circuito cuántico para qPCA
            def create_circuit(data, n_qubits):
                circuit = QuantumCircuit(n_qubits)
                # Añadir puertas de rotación parametrizadas según los datos
                for i in range(n_qubits):
                    circuit.ry(data[i], i)
                return circuit

            n_qubits = quantum_data.shape[1]
            circuits = [create_circuit(row, n_qubits) for row in quantum_data]
            
            # Calcular las expectativas para cada circuito
            expectation_values = []
            for circuit in circuits:
                # Transpilar el circuito para el backend
                transpiled_circuit = transpile(circuit, simulator)
                # Calcular la expectativa del operador Z en el primer qubit
                expectation = estimator.run(transpiled_circuit, observables=['Z' * n_qubits]).result().values
                expectation_values.append(expectation)
                
            # Convertir las expectativas en una matriz numpy
            expectation_matrix = np.array(expectation_values)
            
            # Aplicar una técnica clásica de reducción de dimensiones (por ejemplo, PCA) a las expectativas
            n_samples, n_features = expectation_matrix.shape
            n_components = min(2, min(n_samples, n_features))

            pca = PCA(n_components=n_components)
            reduced_data = pca.fit_transform(expectation_matrix)
        
        elif(circuito_ansatz == 'QAOA'):
            
            def create_graph():
                G = nx.erdos_renyi_graph(12, 0.1)
                return G

            # Crear operador de costo
            def maxcut_operator(graph, num_qubits):
                pauli_list = []
                coeffs = []

                # Para cada arista del grafo, construir un término del Hamiltoniano
                for i, j in graph.edges():
                    # Asegurarse de que los índices estén dentro del rango de los qubits
                    if i >= num_qubits or j >= num_qubits:
                        raise IndexError(f"Índices de los nodos {i} y {j} están fuera del rango de qubits ({num_qubits})")

                    # Crear el pauli string
                    pauli_string = ['I'] * num_qubits

                    # Establecer las posiciones correspondientes de los nodos a 'Z'
                    pauli_string[i], pauli_string[j] = 'Z', 'Z'

                    # Añadir el término al operador
                    pauli_list.append(''.join(pauli_string))
                    coeffs.append(-1.0)  # Coeficiente asociado a la arista

                return SparsePauliOp(pauli_list, coeffs)
            
            # Crear circuito QAOA
            def create_qaoa_circuit(num_qubits, params):
                layers = len(params) // 2
                circuit = QuantumCircuit(num_qubits)

                # Puertas Hadamard
                for qubit in range(num_qubits):
                    circuit.h(qubit)

                # Capas QAOA
                for i in range(layers):
                    gamma, beta = params[2 * i], params[2 * i + 1]
                    for j, k in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
                        circuit.cx(j, k)
                        circuit.rz(2 * gamma, k)
                        circuit.cx(j, k)
                    for qubit in range(num_qubits):
                        circuit.rx(2 * beta, qubit)

                return circuit
            
            # Ejecutar QAOA
            def run_qaoa(graph, p=1):
                num_qubits = graph.number_of_nodes()
                hamiltonian = maxcut_operator(graph, num_qubits)

                # Configurar backend
                simulator = AerSimulator()
                estimator = Estimator(options={"backend": simulator})

                # Inicialización de parámetros
                params = np.random.uniform(0, 2 * np.pi, size=2 * p)

                # Función de costo
                def cost_function(params):
                    circuit = create_qaoa_circuit(num_qubits, params)
                    energy = estimator.run(circuit, hamiltonian).result().values[0]
                    return energy

                # Optimización clásica
                result = minimize(cost_function, params, method="COBYLA", options={"maxiter": 100})
                optimal_params = result.x
                optimal_energy = result.fun

                return optimal_params, optimal_energy
            
            graph = create_graph()
            print("Grafo:", graph.edges())

            # Resolver Max-Cut con QAOA
            p = 2
            params, energy = run_qaoa(graph, p=p)

            print("\nParámetros óptimos:", params)
            print("Energía mínima encontrada:", energy)

    # Agregar la vista inicial a la página
    page.add(login_view)

port = os.getenv("PORT", "8080")
ft.app(target=main, view=ft.WEB_BROWSER, port=port)
