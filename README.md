# TDMS Slack Monitor

Monitors a folder for new TDMS files from quantum sensing experiments, generates diagnostic plots (time series + amplitude spectral density), and posts them to Slack.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cp config.toml myconfig.toml
# Edit myconfig.toml with your settings
python tdms_slack_monitor.py myconfig.toml
```

## Slack setup

1. Create a Slack app at https://api.slack.com/apps
2. Go to **OAuth & Permissions** and add these scopes:
   - `files:write`
   - `chat:write`
3. Install the app to your workspace
4. Copy the **Bot User OAuth Token** (starts with `xoxb-`)
5. Get the channel ID: right-click the channel → View channel details → scroll to bottom
6. Invite the bot to the channel: `/invite @YourBotName`

## Behaviour on startup

The monitor seeds its in-memory state with the 10 most recent files (by mtime) to avoid reprocessing them. New files appearing after startup will be processed and posted.

## Running as a service (systemd)

Create `/etc/systemd/system/tdms-monitor.service`:

```ini
[Unit]
Description=TDMS Slack Monitor
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/monitor
ExecStart=/path/to/python tdms_slack_monitor.py config.toml
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable tdms-monitor
sudo systemctl start tdms-monitor
```
