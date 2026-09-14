Generate the static catalogue and visualization assets from the deployed models:

    python scripts/export_web_assets.py

The first command writes a validated staging export under
training_results_corrected/web_export/. Publish that staged export after review
without rerunning inference:

    python scripts/export_web_assets.py --publish-staged

To generate, validate, and publish in one command instead, use:

    python scripts/export_web_assets.py --publish
