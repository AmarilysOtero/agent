import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  selectedSource: null,
  scanResults: null,
  isLoading: false,
  error: null,
  authStateId: null,
};

const scannerSlice = createSlice({
  name: 'scanner',
  initialState,
  reducers: {
    setSource: (state, action) => {
      state.selectedSource = action.payload;
    },
    setAuthState: (state, action) => {
      state.authStateId = action.payload;
    },
    setScanResults: (state, action) => {
      state.scanResults = action.payload;
    },
    setLoading: (state, action) => {
      state.isLoading = action.payload;
    },
    setError: (state, action) => {
      state.error = action.payload;
    },
    clearResults: (state) => {
      state.scanResults = null;
      state.error = null;
    },
  },
});

export const {
  setSource,
  setAuthState,
  setScanResults,
  setLoading,
  setError,
  clearResults,
} = scannerSlice.actions;

export default scannerSlice.reducer;
