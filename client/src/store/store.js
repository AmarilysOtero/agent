import { configureStore } from '@reduxjs/toolkit';
import scannerReducer from './slices/scannerSlice';

export const store = configureStore({
  reducer: {
    scanner: scannerReducer,
  },
});
