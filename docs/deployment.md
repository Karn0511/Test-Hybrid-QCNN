# Deployment instructions

## Local backend

1. Create a Python 3.10+ virtual environment.
2. Install dependencies from `requirements.txt`.
3. Optionally run `python scripts/setup_environment.py`.
4. Train the model with `python backend/training/train_qcnn.py --prepare-datasets`.
5. Start the API with `uvicorn backend.api.server:app --reload`.

## Angular dashboard

1. Move into `frontend/angular-dashboard/`.
2. Run `npm install`.
3. Update `src/environments/environment.prod.ts` with the deployed backend URL.
4. Run `npm run build`.

## Firebase Hosting

Inside `frontend/angular-dashboard/`:

1. Install Firebase CLI.
2. Replace the project id in `.firebaserc`.
3. Run `firebase deploy --only hosting`.

## Backend container for Cloud Run

Build from the repository root:

- `docker build -f backend/Dockerfile -t ai-qcnn-sentiment-platform .`
- Push the image to Artifact Registry or Container Registry.
- Deploy to Cloud Run and expose port `8000`.
