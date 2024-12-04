from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import streamlit as st
from functools import lru_cache
import plotly.express as px
from PIL import Image
#from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
from concurrent.futures import ThreadPoolExecutor
from joblib import load
import joblib
import mysql.connector
import json
import os
from matplotlib.dates import date2num
import matplotlib.dates as mdates


def main():
    st.set_page_config(page_title="Análisis Climatológico", layout="wide")

    database_password = st.secrets["database_password"]

    def consulta_sql(query):

        database = "AEMET"
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password=st.secrets["database_password"],
            database=database
        )

        cursor = db.cursor()
        cursor.execute(query)

        # Obtiene los datos
        data = cursor.fetchall()
        columns = [col[0] for col in cursor.description]  # Nombres de las columnas
        cursor.close()
        db.close()

        # Convierte los datos en un DataFrame de pandas
        return pd.DataFrame(data, columns=columns)

    menu = ['Inicio', 'Valores climatológicos por comunidad y provincia', 'Comparador de valores climatológicos',
            'Mapa coroplético', 'Predicción del tiempo', 'Facebook Prophet', 'Diagrama MYSQL: Base de datos',
            'About us']

    choice = st.sidebar.selectbox("Selecciona una opción", menu, key="menu_selectbox_unique")

    if choice == "Inicio":
        st.markdown(
            """
            <div style="text-align: center;">
                <img src="https://facuso.es/wp-content/uploads/2023/09/6de3b76f2eeed4e2edfa5420ad9630bd.jpg" 
                     alt="Imagen oficial de la AEMET" 
                     width="250">
                <p>Imagen oficial de la AEMET</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Introducción
        st.markdown(
            "##### Bienvenido/a a la plataforma de exploración de datos de la AEMET (Agencia Estatal Meteorológica de España).")

        st.markdown(
            "##### España, con su diversidad climática y variada geografía, ofrece un rico panorama meteorológico en donde te encontrarás con temperaturas muy dispersas entre ellas en varios puntos geográfico. ")
        st.markdown("A tu izquierda encontrarás varias secciones, en donde cada apartado tendrá una breve introducción")
        st.markdown(
            "En este sitio, podrás explorar y comparar datos históricos desde 2014 para obtener una visión profunda del clima en nuestro país.")

        # Crear 3 columnas para las imágenes
        col1, col2, col3 = st.columns(3)

        # Añadir imágenes a cada columna con un tamaño mayor
        with col1:
            st.image(
                "https://i.pinimg.com/originals/73/93/14/739314e72faa8f68bc12a29dcf0ce07c.jpg",
                caption="Ordesa y Monte Perdido",
                width=250  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://fascinatingspain.com/wp-content/uploads/benasque_nieve.jpg",
                caption="Benasque",
                width=250  # Ajusta el ancho según sea necesario
            )

        with col2:
            st.image(
                "https://www.viajes.com/blog/wp-content/uploads/2021/09/sea-6580532_1920.jpg",
                caption="Galicia, tierra de Meigas",
                width=250  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/cd/14/c8/cd14c8b90c06f714899d0d17e7d7fcd4.jpg",
                caption="Mallorca, Cala Egos - Cala d'Or",
                width=250  # Ajusta el ancho según sea necesario
            )

        with col3:
            st.image(
                "https://palenciaturismo.es/system/files/Monta%C3%B1aPalentinaGaleria5.jpg",
                caption="Palencia",
                width=250  # Ajusta el ancho según sea necesario
            )
            st.image(
                "https://i.pinimg.com/originals/d8/3a/f2/d83af2c8d615f0a8393ef3eeb9325435.jpg",
                caption="Asturias",
                width=250  # Ajusta el ancho según sea necesario
            )

    if choice == "Valores climatológicos por comunidad y provincia":

        # Título de la aplicación
        st.title("🌟 **Análisis Climatológico Interactivo por Ciudad** 🌤️")

        # Descripción de la aplicación
        st.markdown("""
        ### 🏙️ **Explora el Clima en Profundidad** 🌍

        Bienvenido al **Análisis Climatológico Interactivo**, donde podrás descubrir cómo ha evolucionado el clima en diferentes **ciudades** de **España** a lo largo del tiempo. 🌦️

        Selecciona una **ciudad** y un **rango de fechas** para visualizar datos detallados sobre las **temperaturas promedio**, **máximas y mínimas**, **precipitación**, **viento**, **altitud**, **humedad** y más. 📊

        🔍 **¿Qué puedes explorar aquí?**  
        - **Temperaturas media, máxima y mínima** a lo largo del tiempo 🌡️  
        - **Precipitación acumulada** (lluvias y otras condiciones meteorológicas) 🌧️  
        - **Velocidad y dirección del viento** 🌬️  
        - **Altitud** de la ciudad y su relación con el clima ⛰️
        - **Análisis visual** mediante gráficos interactivos 📈

        ¡Comienza a explorar el clima de tu ciudad favorita! 🏙️🌦️
        """)

        # Consulta de las ciudades disponibles
        ciudades_query = "SELECT * FROM ciudades"
        ciudades_df = consulta_sql(ciudades_query)

        # Sección de selección de parámetros
        with st.container():


            col1, col2, col3 = st.columns([1, 1, 1])

            # Sección de selección de parámetros
            with col1:

                ciudad_seleccionada = st.selectbox("🌆 Selecciona una ciudad", ciudades_df['ciudad'].tolist())
                st.write(f"**Datos climáticos para:** {ciudad_seleccionada}")
                ciudad_id = ciudades_df.loc[ciudades_df['ciudad'] == ciudad_seleccionada, 'ciudad_id'].values[0]
                # Obtener provincia_id de la ciudad seleccionada
                ciudad_info_query = f"""
                    SELECT p.provincia_id, c.ciudad_id, c.ciudad
                    FROM ciudades c
                    JOIN provincias p ON c.ciudad_id = p.provincia_id
                    WHERE c.ciudad = '{ciudad_seleccionada}'
                """

                ciudad_info = consulta_sql(ciudad_info_query)

                # Verificar si la consulta devuelve resultados
                if not ciudad_info.empty:
                    provincia_id = ciudad_info['provincia_id'].values[0]
                    ciudad_id = ciudad_info['ciudad_id'].values[0]
                    ciudad = ciudad_info['ciudad'].values[0]
                    st.write(f"**Provincia ID**: {provincia_id} - **Ciudad ID**: {ciudad_id} - **Ciudad**: {ciudad}")
                else:
                    st.warning(f"No se encontraron datos para la ciudad '{ciudad_seleccionada}'.")

                # Obtener nombre de la provincia

                provincia_info_query = f"""
                    SELECT provincia
                    FROM provincias
                    WHERE provincia_id = {provincia_id}
                """
                provincia_info = consulta_sql(provincia_info_query)
                provincia = provincia_info['provincia'].values[0]

                # Mostrar la comunidad y la provincia
                st.write(f"🌍 📍 **Provincia**: {provincia}")

            with col2:
                fecha_inicio = st.date_input("📅 Fecha de inicio:", value=datetime(2014, 1, 1))

            with col3:
                fecha_fin = st.date_input("📅 Fecha de fin:", value=datetime(2024, 10, 31))

        # Consulta de datos climáticos
                query= f"""
                SELECT fecha, tmed, tmax, tmin, prec, velemedia, dir, hrMedia, altitud, hrMax, hrMin
                FROM valores_climatologicos
                WHERE ciudad_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
            """
            datos_climaticos_df = consulta_sql(query)
            st.dataframe(datos_climaticos_df)
        # Comprobación si hay datos
        if not datos_climaticos_df.empty:
            st.subheader("📊 Datos Climáticos")
            st.dataframe(datos_climaticos_df, use_container_width=True)

            # Gráficos interactivos
            with st.container():
                st.markdown("### 🌡️ Visualización General")
                col1, col2 = st.columns(2)

                # Gráfico de Temperatura Media
                with col1:
                    st.markdown("#### 📈 Temperatura Media")
                    fig_temp = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmed'],
                        mode='lines',
                        line=dict(color='red'),
                        name='Temperatura Media'
                    ))
                    fig_temp.update_layout(title="Temperatura Media", xaxis_title="Fecha", yaxis_title="°C")
                    st.plotly_chart(fig_temp, use_container_width=True)

                # Gráfico de Velocidad del Viento
                with col2:
                    st.markdown("#### 💨 Velocidad del Viento")
                    fig_wind = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['velemedia'],
                        mode='lines',
                        line=dict(color='green'),
                        name='Velocidad del Viento'
                    ))
                    fig_wind.update_layout(title="Velocidad Media del Viento", xaxis_title="Fecha", yaxis_title="km/h")
                    st.plotly_chart(fig_wind, use_container_width=True)

            with st.container():
                col3, col4 = st.columns(2)

                # Gráfico de Precipitación
                with col3:
                    st.markdown("#### 🌧️ Precipitación")
                    fig_precip = go.Figure(go.Bar(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['prec'],
                        marker_color='darkblue',
                        name='Precipitación'
                    ))
                    fig_precip.update_layout(title="Precipitación Acumulada", xaxis_title="Fecha", yaxis_title="mm")
                    st.plotly_chart(fig_precip, use_container_width=True)

                # Gráfico de Humedad Relativa
                with col4:
                    st.markdown("#### 💧 Humedad Relativa")
                    fig_humidity = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['hrMedia'],
                        mode='lines',
                        line=dict(color='lightblue'),
                        name='Humedad Relativa'
                    ))
                    fig_humidity.update_layout(title="Humedad Relativa Media", xaxis_title="Fecha", yaxis_title="%")
                    st.plotly_chart(fig_humidity, use_container_width=True)

            with st.container():
                col5, col6 = st.columns(2)

                # Gráfico de Altitud
                with col5:
                    st.markdown("#### ⛰️ Altitud")
                    fig_altitud = go.Figure(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['altitud'],
                        mode='lines',
                        line=dict(color='orange'),
                        name='Altitud'
                    ))
                    fig_altitud.update_layout(title="Altitud de la Ciudad", xaxis_title="Fecha",
                                              yaxis_title="m sobre el nivel del mar")
                    st.plotly_chart(fig_altitud, use_container_width=True)

                # Gráfico de Temperatura Máxima y Mínima
                with col6:
                    st.markdown("#### 🌡️ Temperaturas Máximas y Mínimas")
                    fig_temp_max_min = go.Figure()

                    # Añadir temperatura máxima
                    fig_temp_max_min.add_trace(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmax'],
                        mode='lines',
                        name='Temperatura Máxima',
                        line=dict(color='red', dash='solid')
                    ))

                    # Añadir temperatura mínima
                    fig_temp_max_min.add_trace(go.Scatter(
                        x=datos_climaticos_df['fecha'],
                        y=datos_climaticos_df['tmin'],
                        mode='lines',
                        name='Temperatura Mínima',
                        line=dict(color='blue', dash='solid')
                    ))

                    fig_temp_max_min.update_layout(title="Temperaturas Máximas y Mínimas", xaxis_title="Fecha",
                                                   yaxis_title="°C")
                    st.plotly_chart(fig_temp_max_min, use_container_width=True)

            # Consultas avanzadas (más métricas)
            st.markdown("### 📊 Consultas Avanzadas")
            queries = {
                "Temperaturas Máxima y Mínima Diaria": f"""
                    SELECT fecha, MAX(tmax) AS max_temperature, MIN(tmin) AS min_temperature 
                    FROM valores_climatologicos 
                    WHERE ciudad_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY fecha ORDER BY fecha;
                """,
                "Dirección del Viento Promedio": f"""
                    SELECT fecha, AVG(dir) AS average_wind_direction
                    FROM valores_climatologicos
                    WHERE ciudad_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY fecha ORDER BY fecha;
                """,
                "Precipitación Total Mensual": f"""
                    SELECT DATE_FORMAT(fecha, '%Y-%m') AS month, SUM(prec) AS total_precipitation 
                    FROM valores_climatologicos 
                    WHERE ciudad_id = {ciudad_id} AND fecha BETWEEN '{fecha_inicio}' AND '{fecha_fin}'
                    GROUP BY month ORDER BY month;
                """,
            }

            selected_query = st.selectbox("🔍 Selecciona una consulta avanzada:", list(queries.keys()))
            data_avanzada = consulta_sql(queries[selected_query])

            if not data_avanzada.empty:
                fig_advanced = go.Figure()
                for col in data_avanzada.columns[1:]:
                    fig_advanced.add_trace(go.Scatter(
                        x=data_avanzada['fecha'],
                        y=data_avanzada[col],
                        mode='lines',
                        name=col
                    ))
                fig_advanced.update_layout(title=f"Análisis Avanzado - {selected_query}", xaxis_title="Fecha",
                                           yaxis_title="Valor")
                st.plotly_chart(fig_advanced, use_container_width=True)
        else:
            st.warning("⚠️ No se encontraron datos para el rango de fechas seleccionado.")

    if choice == "Facebook Prophet":


        # Configuración de la página
        st.title("Predicciones Climatológicas")

        # Títulos de la aplicación
        st.subheader("Modelos Predictivos de Facebook Prophet")
        st.write("Cargue modelos preentrenados para realizar predicciones sobre los datos reales.")


        # Función para cargar modelos .pkl
        def load_model(file_path):
            return load(file_path)

        # Carga de modelos
        models = {
            "Modelo Semestral": load_model("prophet_biannual.pkl"),
            "Modelo Trimestral": load_model("prophet_quarterly.pkl"),
            "Modelo Mensual": load_model("prophet_monthly.pkl"),
            "Modelo Semanal": load_model("prophet_weekly.pkl"),
            "Modelo Diario": load_model("prophet_daily.pkl")
        }

        # Input de datos reales (simulamos que provienen de una query)
        query1 = """
            SELECT fecha, tmed
            FROM valores_climatologicos
             """

        # Simulación de datos obtenidos (reemplaza esto por tu extracción real de datos)
        data_real = consulta_sql(query1)
        st.image("https://estaticos-cdn.prensaiberica.es/clip/c086f7c2-e053-4e0a-889e-8bbb4f55197f_16-9-discover-aspect-ratio_default_0.webp",
                caption="Temperaturas España",
                width=600  # Ajusta el ancho según sea necesario
        )

        # Conversión al formato requerido por Prophet
        data_real.rename(columns={"fecha": "ds", "tmed": "y"}, inplace=True)

        # Selección del modelo
        model_choice = st.selectbox(
            "Seleccione el modelo que desee utilizar:",
            list(models.keys())
        )

        times = {"Mañana" : 1,
                 "Semana" : 7,
                 "Quincena" : 14,
                 "Mes" : 30}

        times_choice = st.selectbox("Seleccione el rango de tiempo que desee predecir:",
                                    list(times.keys()))

        st.write("**Predicción de la temperatura media de mañana según los distintos modelos.**",
                 )

        # Predicción con el modelo seleccionado
        if st.button("Predecir"):

            model = models[model_choice]  # Asegúrate de que este es un modelo Prophet
            future = model.make_future_dataframe(periods=times[times_choice], freq='D')
            forecast = model.predict(future)
            st.write(f"**Temperatura para {times_choice} mediante el modelo  {model_choice}:**")
            forecast_reset = forecast[['ds', 'yhat']].tail(times[times_choice]).reset_index(drop=True)
            forecast_reset.index = range(1, len(forecast_reset) + 1)
            forecast_reset['yhat'] = forecast_reset['yhat'].round(2).astype(str) + " ºC"
            forecast_reset['ds'] = forecast_reset['ds'].dt.date
            forecast_reset = forecast_reset.rename(columns={"ds": "Fecha", "yhat": "Temperatura media"})
            st.dataframe(forecast_reset[['Fecha', 'Temperatura media']].tail(times[times_choice]))

            st.write("**Gráfico de predicciones:**")

            # Crear el gráfico original
            fig = model.plot(forecast)

            # Obtener el eje actual del gráfico
            ax = fig.gca()

            # Definir la fecha límite entre datos reales y predicciones
            cutoff_date = data_real['ds'].max()

            # Añadir una línea vertical para marcar la separación
            ax.axvline(x=date2num(cutoff_date), color='red', linestyle='--', label='Fin datos reales')

            # Sombrear la zona de datos reales
            ax.axvspan(ax.get_xlim()[0], date2num(cutoff_date), color='lightblue', alpha=0.3, label='Datos reales')

            # Sombrear la zona de predicciones
            ax.axvspan(date2num(cutoff_date), ax.get_xlim()[1], color='lightgreen', alpha=0.3, label='Predicciones')


            # Agregar título y etiquetas
            ax.set_title(f"Predicción para {times_choice}")
            ax.set_xlabel("Años")
            ax.set_ylabel("Temperatura (ºC)")
            ax.legend()

            ax.legend()

            # Mostrar el gráfico original en Streamlit
            st.pyplot(fig)



            # GRÁFICO ZOOM: Crear un segundo gráfico para el zoom
            fig_zoom, ax_zoom = plt.subplots()

            # Convertir 'ds' a tipo datetime si es necesario
            forecast['ds'] = pd.to_datetime(forecast['ds'])

            # Asegurar que cutoff_date es de tipo datetime
            cutoff_date = pd.to_datetime(cutoff_date)

            # Filtrar los datos
            zoom_data = forecast[forecast['ds'] >= cutoff_date]

            # Graficar los datos con un enfoque en la zona de predicción
            ax_zoom.plot(forecast['ds'], forecast['yhat'], label='Predicción')
            ax_zoom.fill_between(
                forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                color='gray', alpha=0.2, label='Intervalo de confianza'
            )

            # Añadir la línea vertical y sombreado
            ax_zoom.axvline(x=date2num(cutoff_date), color='red', linestyle='--', label='Fin datos reales')
            ax_zoom.axvspan(date2num(cutoff_date), ax_zoom.get_xlim()[1], color='lightgreen', alpha=0.3,
                            label='Predicciones')

            # Configurar límites del eje X para enfocar en la zona de interés
            ax_zoom.set_xlim([cutoff_date, forecast['ds'].max()])

            # Si las fechas son iguales, muestra solo un valor en el eje X
            if len(forecast['ds'].unique()) == 1:  # Si todas las fechas son iguales
                ax_zoom.set_xticks([forecast['ds'].min()])  # Solo una etiqueta
                ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de fecha
            else:
                ax_zoom.xaxis.set_major_locator(mdates.DayLocator())  # Para un rango mayor, usar DayLocator o similar
                ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de fecha

            # Ajustar la rotación de las etiquetas del eje X
            fig_zoom.autofmt_xdate(rotation=45)  # Cambiar el ángulo de rotación

            # Agregar título y etiquetas
            ax_zoom.set_title(f"Zoom en la zona de predicción para {times_choice}")
            ax_zoom.set_xlabel("Fecha")
            ax_zoom.set_ylabel("Temperatura (ºC)")
            ax_zoom.legend()

            # Mostrar el gráfico de zoom en Streamlit
            st.pyplot(fig_zoom)
    if choice=="Predicción del tiempo":

        scaler = joblib.load('scaler.pkl')

        # Función para cargar el modelo
        @st.cache_resource
        def load_model_(model_path):
            return load_model(model_path)

        # Interfaz de usuario de Streamlit
        st.title('Predicción de la Temperatura')
        # Entrada de usuario
        estimated_temperature_tomorrow = st.number_input('Temperatura estimada de mañana:', value=27)

        # Cargar el modelo al iniciar la aplicación
        model_gru = load_model_('mi_modelo.keras')

        # Función para predecir la temperatura
        def predict_temperature(model, input_data):
            input_data_scaled = scaler.transform(input_data.reshape(1, -1))
            predicted_temperature = model.predict(input_data_scaled)
            return scaler.inverse_transform(predicted_temperature.reshape(-1, 1))

        # Predecir datos con el modelo GRU
        if st.button('Predecir temperatura'):
            num_features = 1  # Cambia esto según el número de características de tu modelo
            input_data = np.array([[estimated_temperature_tomorrow] * num_features]).reshape(1, num_features)
            # Realizar la predicción
            try:
                predicted_temperature = predict_temperature(model_gru, input_data)
                st.success(f'Predicción de temperatura: {predicted_temperature[0][0]:.2f} °C')
            except Exception as e:
                st.error(f'Ocurrió un error: {e}')

    if choice == "Mapa coroplético":
        st.title("📊 Mapa Coroplético: Histórico de Temperaturas Medias en España")
        st.subheader("Explora las temperaturas medias de España con filtros dinámicos.")
        st.info("""
            1. Filtra por años, meses y provincias para ver las temperaturas medias en cada provincia.
            2. Un **mapa coroplético** es una representación geográfica en la que las áreas del mapa se colorean según valores de una variable. En este caso, se visualizan las **temperaturas medias** de cada provincia, lo que te permite identificar patrones geográficos de temperatura a lo largo del tiempo.
            3. Puedes interactuar con el mapa, seleccionar diferentes fechas y provincias para obtener información precisa y detallada sobre el clima en cada región.
        """)

        def crear_mapa_choropleth(geojson, df, color_column, fill_color, legend_name):
            mapa = folium.Map(location=[40.4168, -3.7038], zoom_start=6)
            folium.Choropleth(
                geo_data=geojson,
                data=df,
                columns=["provincia", color_column],
                key_on="feature.properties.name",
                fill_color=fill_color,  # Colores según la temperatura
                fill_opacity=0.7,
                line_opacity=0.6,
                legend_name=legend_name,
            ).add_to(mapa)
            return mapa



        year = st.selectbox("Selecciona el año:", [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        month = st.selectbox("Selecciona el mes:", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        provincia_seek = st.selectbox("Selecciona la provincia:", [
            'STA. CRUZ DE TENERIFE', 'BARCELONA', 'SEVILLA', 'CUENCA', 'ZARAGOZA', 'ILLES BALEARS', 'VALENCIA',
            'ZAMORA', 'PALENCIA', 'CASTELLON', 'LAS PALMAS', 'MADRID', 'CANTABRIA', 'GRANADA', 'TERUEL', 'BADAJOZ',
            'A CORUÑA', 'ASTURIAS', 'TARRAGONA', 'ALMERIA', 'ALICANTE', 'CADIZ', 'TOLEDO', 'BURGOS', 'GIRONA', 'MALAGA',
            'JAEN', 'MURCIA', 'LLEIDA', 'HUESCA', 'ALBACETE', 'NAVARRA', 'CORDOBA', 'OURENSE', 'CIUDAD REAL',
            'GIPUZKOA',
            'MELILLA', 'LEON', 'CACERES', 'SALAMANCA', 'HUELVA', 'LA RIOJA', 'BIZKAIA', 'GUADALAJARA', 'VALLADOLID',
            'ARABA/ALAVA', 'PONTEVEDRA', 'SEGOVIA', 'SORIA', 'AVILA', 'CEUTA', 'LUGO', 'BALEARES'
        ])

        # Query para obtener los datos filtrados
        query = f"""
            SELECT 
                DATE_FORMAT(t1.fecha, '%Y-%m') AS mes, 
                ROUND(AVG(t1.tmed), 2) AS media_tmed_mensual,
                t1.provincia_id, 
                t2.provincia 
            FROM 
                valores_climatologicos t1 
            RIGHT JOIN 
                provincias t2 ON t1.provincia_id = t2.provincia_id
            WHERE
                YEAR(t1.fecha) = {year}
                AND MONTH(t1.fecha) = {month}
                AND t2.provincia = '{provincia_seek}'
            GROUP BY 
                mes, t1.provincia_id, t2.provincia;
        """
        df = consulta_sql(query)

        # Cargar el archivo GeoJSON de las provincias españolas
        with open("spain-provinces.geojson", "r", encoding="utf-8") as file:
            geojson_spain = json.load(file)

        # Mapeo de nombres de provincias
        map_provincia = {
            "STA. CRUZ DE TENERIFE": "Santa Cruz De Tenerife", "BARCELONA": "Barcelona",
            "SEVILLA": "Sevilla", "CUENCA": "Cuenca", "ZARAGOZA": "Zaragoza", "ILLES BALEARS": "Illes Balears",
            'VALENCIA': "València/Valencia", 'ZAMORA': "Zamora", 'PALENCIA': "Palencia",
            'CASTELLON': "Castelló/Castellón", 'LAS PALMAS': "Las Palmas", 'MADRID': "Madrid",
            'CANTABRIA': "Cantabria", 'GRANADA': "Granada", 'TERUEL': "Teruel", 'BADAJOZ': "Badajoz",
            'A CORUÑA': "A Coruña", 'ASTURIAS': "Asturias", 'TARRAGONA': "Tarragona", 'ALMERIA': "Almería",
            'ALICANTE': "Alacant/Alicante", 'CADIZ': "Cádiz", 'TOLEDO': "Toledo", 'BURGOS': "Burgos",
            'GIRONA': "Girona", 'MALAGA': "Málaga", 'JAEN': "Jaén", 'MURCIA': "Murcia", 'LLEIDA': "Lleida",
            'HUESCA': "Huesca", 'ALBACETE': "Albacete", 'NAVARRA': "Navarra", 'CORDOBA': "Córdoba",
            'OURENSE': "Ourense", 'CIUDAD REAL': "Ciudad Real", 'GIPUZKOA': "Gipuzkoa/Guipúzcoa", 'MELILLA': "Melilla",
            'LEON': "León", 'CACERES': "Cáceres", 'SALAMANCA': "Salamanca", 'HUELVA': "Huelva",
            'LA RIOJA': "La Rioja", 'BIZKAIA': "Bizkaia/Vizcaya", 'GUADALAJARA': "Guadalajara",
            'VALLADOLID': "Valladolid", 'ARABA/ALAVA': "Araba/Álava", 'PONTEVEDRA': "Pontevedra",
            'SEGOVIA': "Segovia", 'SORIA': "Soria", 'AVILA': "Ávila", 'CEUTA': "Ceuta", 'LUGO': "Lugo",
            'BALEARES': "Illes Balears"
        }
        df["provincia"] = df["provincia"].map(map_provincia)

        # Visualizar la tabla con datos de temperaturas
        st.markdown("### Datos de temperaturas medias mensuales:")
        st.dataframe(df)

        # Crear y mostrar el mapa choropleth
        mapa_espana = crear_mapa_choropleth(geojson_spain, df, "media_tmed_mensual", "YlGnBu",
                                            "Temperatura Media Mensual (°C)")
        st_folium(mapa_espana, width=725)

        def mostrar_grafico_temperatura(df, provincia, year, month):
            fig = px.bar(
                df,
                x="mes",
                y="media_tmed_mensual",
                title=f"Tendencia de Temperatura Media en {provincia} ({month}/{year})",
                labels={"mes": "Mes", "media_tmed_mensual": "Temperatura Media (°C)"},
                text="media_tmed_mensual"
            )

            # Reducir el ancho de las barras
            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside', width=0.4)

            # Ajustar tamaño del gráfico y eliminar los textos del eje X
            fig.update_layout(
                width=400,  # Ancho del gráfico
                height=500,  # Altura del gráfico
                title=dict(font=dict(size=13)),  # Tamaño de fuente del título
                xaxis=dict(
                    showticklabels=False,  # No mostrar las etiquetas en el eje X
                    showgrid= False  # Eliminar las líneas de la cuadrícula del eje X
                ),
                yaxis=dict(
                    showgrid=True  # Mostrar líneas de cuadrícula en el eje Y si lo deseas
                )
            )

            st.plotly_chart(fig)

        # Gráfico de tendencia de temperatura
        st.markdown("### Gráfico de Temperaturas Medias Mensuales:")
        mostrar_grafico_temperatura(df, provincia_seek, year, month)

        # Información adicional sobre la fecha seleccionada
        st.info("2. Mapa de España con Temperaturas Medias Diarias.")

        # Selección de fecha
        date = st.date_input("Selecciona una fecha", value=pd.to_datetime(f"2023-01-01"))
        dia = date.strftime('%Y-%m-%d')

        # Consulta para el clima en la fecha seleccionada
        query1 = f"""
            SELECT 
                t1.fecha, 
                ROUND(AVG(t1.tmed), 2) AS media_tmed, 
                t1.provincia_id, 
                t2.provincia 
            FROM 
                valores_climatologicos t1 
            RIGHT JOIN 
                provincias t2 ON t1.provincia_id = t2.provincia_id
            WHERE
                t1.fecha = "{date}"
            GROUP BY 
                t1.fecha, t1.provincia_id, t2.provincia;
        """
        df_daily = consulta_sql(query1)
        df_daily = df_daily[["fecha", "media_tmed", "provincia"]]
        df_daily["provincia"] = df_daily["provincia"].map(map_provincia)

        # Mostrar el gráfico interactivo para la fecha seleccionada
        st.write(f"### Temperatura Media para el día {dia}:")
        st.dataframe(df_daily)

        # Mapa interactivo de temperaturas diarias
        mapa_espana_daily = crear_mapa_choropleth(geojson_spain, df_daily, "media_tmed", "YlOrRd",
                                                  "Temperatura Media Diaria (°C)")
        st_folium(mapa_espana_daily, width=725)

    if choice == "Predicción del tiempo":

        st.title("Predicción de Temperatura para mañana")

        def cargar_modelos_y_escaladores():
            # Definir las carpetas para los modelos y los escaladores de cada tipo
            modelos = {'GRU': {}, 'RNN': {}, 'LSTM': {}}
            escaladores = {'GRU': {}, 'RNN': {}, 'LSTM': {}}

            # Directorios de los modelos y escaladores
            carpeta_gru_modelos = 'modelos_GRU'
            carpeta_rnn_modelos = 'modelos_RNN'
            carpeta_lstm_modelos = 'modelos_LSTM'
            carpeta_gru_scalers = 'scalers_GRU'
            carpeta_rnn_scalers = 'scalers_RNN'
            carpeta_lstm_scalers = 'scalers_LSTM'

            # Cargar modelos GRU
            for archivo in os.listdir(carpeta_gru_modelos):
                if archivo.endswith('.pkl') and 'modelo' in archivo:
                    indicativo = archivo.split('_')[1]
                    modelos['GRU'][indicativo] = joblib.load(os.path.join(carpeta_gru_modelos, archivo))

            # Cargar modelos RNN
            for archivo in os.listdir(carpeta_rnn_modelos):
                if archivo.endswith('.pkl') and 'modelo' in archivo:
                    indicativo = archivo.split('_')[1]
                    modelos['RNN'][indicativo] = joblib.load(os.path.join(carpeta_rnn_modelos, archivo))

            # Cargar modelos LSTM
            for archivo in os.listdir(carpeta_lstm_modelos):
                if archivo.endswith('.pkl') and 'modelo' in archivo:
                    indicativo = archivo.split('_')[1]
                    modelos['LSTM'][indicativo] = joblib.load(os.path.join(carpeta_lstm_modelos, archivo))

            # Cargar escaladores GRU
            for archivo in os.listdir(carpeta_gru_scalers):
                if archivo.endswith('.pkl') and 'scaler' in archivo:
                    indicativo = archivo.split('_')[1]
                    escaladores['GRU'][indicativo] = joblib.load(os.path.join(carpeta_gru_scalers, archivo))

            # Cargar escaladores RNN
            for archivo in os.listdir(carpeta_rnn_scalers):
                if archivo.endswith('.pkl') and 'scaler' in archivo:
                    indicativo = archivo.split('_')[1]
                    escaladores['RNN'][indicativo] = joblib.load(os.path.join(carpeta_rnn_scalers, archivo))

            # Cargar escaladores LSTM
            for archivo in os.listdir(carpeta_lstm_scalers):
                if archivo.endswith('.pkl') and 'scaler' in archivo:
                    indicativo = archivo.split('_')[1]
                    escaladores['LSTM'][indicativo] = joblib.load(os.path.join(carpeta_lstm_scalers, archivo))

            return modelos, escaladores

        # Función para hacer la predicción
        def predecir(model, scaler, datos_entrada):
            # Escalar los datos de entrada
            datos_entrada_scaled = scaler.transform([datos_entrada])
            # Realizar la predicción
            prediccion = model.predict(datos_entrada_scaled)
            return prediccion[0]

        # Streamlit UI
        st.title('Predicción de Temperatura para el Siguiente Día')

        # Ingreso de datos para hacer la predicción
        st.header('Ingresa los datos para la predicción')

        # Entradas del usuario
        tmed = st.number_input('Temperatura media actual (tmed)', value=25.0)
        prec = st.number_input('Precipitación (prec)', value=0.0)
        tmin = st.number_input('Temperatura mínima (tmin)', value=18.0)
        tmax = st.number_input('Temperatura máxima (tmax)', value=30.0)
        dir = st.number_input('Dirección del viento (dir)', value=180)
        velemedia = st.number_input('Velocidad media del viento (velemedia)', value=5.0)
        hrMedia = st.number_input('Humedad relativa media (hrMedia)', value=60.0)
        hrMax = st.number_input('Humedad relativa máxima (hrMax)', value=80.0)

        # Botón para realizar la predicción
        if st.button('Predecir Temperatura para Mañana'):
            # Cargar modelos y escaladores desde las carpetas correspondientes
            modelos, escaladores = cargar_modelos_y_escaladores()

            # Si no hay modelos disponibles, mostrar mensaje de error
            if not modelos:
                st.error("No se encontraron modelos disponibles.")
            else:
                # Seleccionar el tipo de modelo (GRU, RNN, LSTM)
                tipo_modelo = st.selectbox('Selecciona el tipo de modelo', ['GRU', 'RNN', 'LSTM'])

                # Seleccionar el indicativo
                indicativos = list(modelos[tipo_modelo].keys())
                indicativo = st.selectbox('Selecciona el Indicativo', indicativos)

                # Cargar el modelo y el escalador correspondientes al indicativo seleccionado
                model = modelos[tipo_modelo][indicativo]
                scaler = escaladores[tipo_modelo][indicativo]

                # Crear un arreglo con los datos ingresados
                datos_entrada = np.array([tmed, prec, tmin, tmax, dir, velemedia, hrMedia, hrMax])

                # Realizar la predicción
                prediccion = predecir(model, scaler, datos_entrada)

                # Mostrar la predicción
                st.write(f'La predicción de temperatura media para mañana es: {prediccion:.2f} °C')

    if choice == "Facebook Prophet":

        # Configuración de la página
        st.title("Predicciones Climatológicas")

        # Títulos de la aplicación
        st.subheader("Modelos Predictivos de Facebook Prophet")
        st.write("Cargue modelos preentrenados para realizar predicciones sobre los datos reales.")

        # Función para cargar modelos .pkl
        def load_model(file_path):
            return load(file_path)

        # Carga de modelos
        models = {
            "Modelo Semestral": load_model("prophet_biannual.pkl"),
            "Modelo Trimestral": load_model("prophet_quarterly.pkl"),
            "Modelo Mensual": load_model("prophet_monthly.pkl"),
            "Modelo Semanal": load_model("prophet_weekly.pkl"),
            "Modelo Diario": load_model("prophet_daily.pkl")
        }

        # Input de datos reales (simulamos que provienen de una query)
        query1 = """
               SELECT fecha, tmed
               FROM valores_climatologicos
                """

        # Simulación de datos obtenidos (reemplaza esto por tu extracción real de datos)
        data_real = consulta_sql(query1)
        st.image(
            "https://estaticos-cdn.prensaiberica.es/clip/c086f7c2-e053-4e0a-889e-8bbb4f55197f_16-9-discover-aspect-ratio_default_0.webp",
            caption="Temperaturas España",
            width=600  # Ajusta el ancho según sea necesario
            )

        # Conversión al formato requerido por Prophet
        data_real.rename(columns={"fecha": "ds", "tmed": "y"}, inplace=True)

        # Selección del modelo
        model_choice = st.selectbox(
            "Seleccione el modelo que desee utilizar:",
            list(models.keys())
        )

        times = {"Mañana": 1,
                 "Semana": 7,
                 "Quincena": 14,
                 "Mes": 30}

        times_choice = st.selectbox("Seleccione el rango de tiempo que desee predecir:",
                                    list(times.keys()))

        st.write("**Predicción de la temperatura media de mañana según los distintos modelos.**",
                 )

        # Predicción con el modelo seleccionado
        if st.button("Predecir"):

            model = models[model_choice]  # Asegúrate de que este es un modelo Prophet
            future = model.make_future_dataframe(periods=times[times_choice], freq='D')
            forecast = model.predict(future)
            st.write(f"**Temperatura para {times_choice} mediante el modelo  {model_choice}:**")
            forecast_reset = forecast[['ds', 'yhat']].tail(times[times_choice]).reset_index(drop=True)
            forecast_reset.index = range(1, len(forecast_reset) + 1)
            forecast_reset['yhat'] = forecast_reset['yhat'].round(2).astype(str) + " ºC"
            forecast_reset['ds'] = forecast_reset['ds'].dt.date
            forecast_reset = forecast_reset.rename(columns={"ds": "Fecha", "yhat": "Temperatura media"})
            st.dataframe(forecast_reset[['Fecha', 'Temperatura media']].tail(times[times_choice]))

            st.write("**Gráfico de predicciones:**")

            # Crear el gráfico original
            fig = model.plot(forecast)

            # Obtener el eje actual del gráfico
            ax = fig.gca()

            # Definir la fecha límite entre datos reales y predicciones
            cutoff_date = data_real['ds'].max()

            # Añadir una línea vertical para marcar la separación
            ax.axvline(x=date2num(cutoff_date), color='red', linestyle='--', label='Fin datos reales')

            # Sombrear la zona de datos reales
            ax.axvspan(ax.get_xlim()[0], date2num(cutoff_date), color='lightblue', alpha=0.3, label='Datos reales')

            # Sombrear la zona de predicciones
            ax.axvspan(date2num(cutoff_date), ax.get_xlim()[1], color='lightgreen', alpha=0.3, label='Predicciones')

            # Agregar título y etiquetas
            ax.set_title(f"Predicción para {times_choice}")
            ax.set_xlabel("Años")
            ax.set_ylabel("Temperatura (ºC)")
            ax.legend()

            ax.legend()

            # Mostrar el gráfico original en Streamlit
            st.pyplot(fig)

            # GRÁFICO ZOOM: Crear un segundo gráfico para el zoom
            fig_zoom, ax_zoom = plt.subplots()

            # Convertir 'ds' a tipo datetime si es necesario
            forecast['ds'] = pd.to_datetime(forecast['ds'])

            # Asegurar que cutoff_date es de tipo datetime
            cutoff_date = pd.to_datetime(cutoff_date)

            # Filtrar los datos
            zoom_data = forecast[forecast['ds'] >= cutoff_date]

            # Graficar los datos con un enfoque en la zona de predicción
            ax_zoom.plot(forecast['ds'], forecast['yhat'], label='Predicción')
            ax_zoom.fill_between(
                forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                color='gray', alpha=0.2, label='Intervalo de confianza'
            )

            # Añadir la línea vertical y sombreado
            ax_zoom.axvline(x=date2num(cutoff_date), color='red', linestyle='--', label='Fin datos reales')
            ax_zoom.axvspan(date2num(cutoff_date), ax_zoom.get_xlim()[1], color='lightgreen', alpha=0.3,
                            label='Predicciones')

            # Configurar límites del eje X para enfocar en la zona de interés
            ax_zoom.set_xlim([cutoff_date, forecast['ds'].max()])

            # Si las fechas son iguales, muestra solo un valor en el eje X
            if len(forecast['ds'].unique()) == 1:  # Si todas las fechas son iguales
                ax_zoom.set_xticks([forecast['ds'].min()])  # Solo una etiqueta
                ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de fecha
            else:
                ax_zoom.xaxis.set_major_locator(mdates.DayLocator())  # Para un rango mayor, usar DayLocator o similar
                ax_zoom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Formato de fecha

            # Ajustar la rotación de las etiquetas del eje X
            fig_zoom.autofmt_xdate(rotation=45)  # Cambiar el ángulo de rotación

            # Agregar título y etiquetas
            ax_zoom.set_title(f"Zoom en la zona de predicción para {times_choice}")
            ax_zoom.set_xlabel("Fecha")
            ax_zoom.set_ylabel("Temperatura (ºC)")
            ax_zoom.legend()

            # Mostrar el gráfico de zoom en Streamlit
            st.pyplot(fig_zoom)
    if choice == "Diagrama MYSQL: Base de datos":
        st.image(image="Esquema_AEMET.png",
                 caption="Esquema de la base de datos AEMET",
                 use_column_width=True)

        st.subheader("Esquema base de datos AEMET:")
        st.write("""El esquema de esta base de datos consta de 4 tablas de datos en la que la principal sería la tabla llamada valores climatológicos y de la que surgen otras tres tablas llamadas indicativo, ciudades y provincias."
                                "En la tabla principal podemos encontrar los siguientes datos:

                   Fecha: recoge la fecha de medición de los valores climatológicos.

                   Altitud: altitud de medición de estos valores.

                   Tmed: temperatura media recogida durante el día en grados centígrados.

                   Prec: precipitaciones acumuladas en milímetros, que equivale a un 1 litro de agua por metro cuadrado."

                   Tmin: temperatura mínima registrada en el día.

                   HoraTmin: registro de hora de temperatura mínima.

                   Tmax: temperatura máxima registrada en el día.

                   HoraTmax: registro de hora de temperatura máxima.

                   Dir: direccional predominante del viento, expresada en grados (0°-360°) o en puntos cardinales (N, NE, E, etc.). Esto señala de dónde viene el viento, no hacia dónde va.

                   Velemedia: se refiere a la velocidad media del viento, expresada generalmente en kilómetros por hora (km/h) o metros por segundo (m/s). Este valor representa la velocidad promedio del viento registrada en el día.

                   Racha: se refiere a la racha máxima de viento, que es la mayor velocidad instantánea del viento registrada en un periodo determinado.

                   Horaracha: registro de hora de Racha.

                   HrMedia: Humedad relativa media del día.

                   HrMax: Humedad máxima registrada en el día.

                   HoraHrMax: Hora de registro de la humedad máxima.

                   HrMin: Humedad mínima registrada en el día.

                   HoraHrMin: Hora de registro de la humedad mínima.

                   Indicativo_id: índice asignado al valor indicativo de estación meteorológica.

                   Ciudad_id: índice asignado al valor ciudad.

                   Provincia_id: índice asignado al valor provincia.""")

    if choice == "About us":
        st.title("📬 **Contacto y Desarrolladores**")
        st.subheader(
            "Este proyecto ha sido desarrollado por los alumnos del curso de Data Science & IA. A continuación, encontrarás los datos de contacto.")

        # Establecer el tamaño de las imágenes (más pequeñas para un diseño más elegante)
        size = (250, 250)  # Imagen más pequeña y profesional

        # Cargar las imágenes de los miembros del equipo
        estela_img = Image.open("Estela.jpeg").resize(size)
        pablo_img = Image.open("Pablo Petidier.jpeg").resize(size)

        # Crear dos columnas
        col1, col2 = st.columns(2)

        # Primera columna (Estela)
        with col1:
            st.image(estela_img, caption="Estela Mojena Ávila", use_column_width=False)
            st.markdown("**Estela Mojena Ávila**")
            st.markdown("**📧 Correo Electrónico:** [estelamojenaavila@gmail.com](mailto:estelamojenaavila@gmail.com)")
            st.markdown("**📞 Teléfono:** [+34 622 68 33 95](tel:+34622683395)")
            st.markdown("**💼 LinkedIn:** [Estela Mojena Ávila](https://www.linkedin.com/in/estela-mojena-avila/)")
            st.markdown("**💻 GitHub:** [Estela8](https://github.com/Estela8)")

        # Segunda columna (Pablo)
        with col2:
            st.image(pablo_img, caption="Pablo Petidier Smit", use_column_width=False)
            st.markdown("**Pablo Petidier Smit**")
            st.markdown("**📧 Correo Electrónico:** [petidiersmit@gmail.com](mailto:petidiersmit@gmail.com)")
            st.markdown("**📞 Teléfono:** [+34 624 10 85 03](tel:+34624108503)")
            st.markdown("**💼 LinkedIn:** [Pablo Petidier Smit](https://www.linkedin.com/in/pablopetidier/)")
            st.markdown("**💻 GitHub:** [ppswns1988](https://github.com/ppswns1988)")

        # Espacio adicional para separar la información
        st.markdown("---")

        # Descripción del proyecto de manera breve
        st.markdown("""
        **Descripción del Proyecto:**  
        Este proyecto ha sido desarrollado como parte del curso de Data Science & IA. Su objetivo es proporcionar un análisis interactivo y visual de datos climáticos históricos, permitiendo a los usuarios explorar el clima de diferentes ciudades y provincias a lo largo del tiempo.

        **Objetivos del Proyecto:**  
        - Visualización de datos climáticos históricos (temperaturas, precipitación, viento, humedad).
        - Provisión de herramientas de análisis interactivo para el usuario.
        - Desarrollo y despliegue de un proyecto basado en Python y Streamlit.

        **Tecnologías Utilizadas:**  
        - **Python**: para procesamiento de datos.
        - **Streamlit**: para la creación de interfaces web interactivas.
        - **Plotly**: para gráficos interactivos.
        - **MySQL**: como base de datos para almacenar la información climática.

        **Fecha de Creación:** Octubre 2024
        """)

        st.markdown("---")

        # Agradecimiento y Cierre
        st.markdown("""
        **Agradecimientos:**  
        Agradecemos el apoyo recibido durante el curso de Data Science & IA, así como a todos aquellos que contribuyeron al desarrollo y mejora de este proyecto.

        Si tienes alguna duda o deseas ponerte en contacto con nosotros, no dudes en escribirnos a través de los correos electrónicos proporcionados.

        """)


if __name__ == "__main__":
    main()