# Trading Bot EU+US (Alpaca + IBKR/Simulatore)

Bot di trading automatico con:
- Mercato USA (via Alpaca API)
- Mercato EU (via IBKR o simulazione Yahoo)
- Report giornalieri su Telegram
- Auto-learning dei parametri strategici

## Deploy su Render
1. Crea una repo GitHub con i file: bot.py, requirements.txt, .env.example, README.md
2. Su [Render](https://render.com) crea un nuovo Web Service collegando la repo
3. Inserisci le *Environment Variables* da .env.example con i tuoi valori reali
4. Start Command: python bot.py
5. Deploy: il bot gira 24/7 e invia messaggi su Telegram
