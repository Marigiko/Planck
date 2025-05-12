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
