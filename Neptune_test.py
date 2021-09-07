# Notebook de prueba para conectar Neptune.ai
import neptune.new as neptune # Cargar Neuptune

run = neptune.init(project='Username/ProjectName',api_token='Personal API key') # Las credenciales (usuario, nombre del proyecto, llave personal)

run["JIRA"] = "NPT-952"
run["parameters"] = {"learning_rate": 0.001,
                     "optimizer": "Adam"}

for epoch in range(100):
   run["train/loss"].log(epoch * 0.4)
run["eval/f1_score"] = 0.66

import ssl
neptune.stop()
