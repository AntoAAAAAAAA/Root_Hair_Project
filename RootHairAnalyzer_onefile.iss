; Root Hair Analyzer - Inno Setup installer (PyInstaller --onefile)

#define MyAppName "RootHairAnalyzer"
#define MyAppVersion "2.0"
#define MyAppPublisher "Discovering Signals Stream"
#define MyAppURL "https://github.com/AntoAAAAAAAA/Root_Hair_Project.git"
#define MyAppExeName "RootHairAnalyzer.exe"

[Setup]
AppId={{A1C6ADC5-6D88-49AC-B2D6-8814485CFF3D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\Root Hair Analyzer
DefaultGroupName=Root Hair Analyzer
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=output
OutputBaseFilename=RHA_Installer
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The PyInstaller executable must be genuinely standalone for this version to work.
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\Root Hair Analyzer"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\Root Hair Analyzer"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Root Hair Analyzer"; Flags: nowait postinstall skipifsilent
