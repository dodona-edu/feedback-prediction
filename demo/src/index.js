import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import 'bootstrap/dist/css/bootstrap.css'
import App from './routes/App';
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import Review from "./routes/review";

const router = createBrowserRouter([
    {
        path: "/",
        element: <App/>
    },
    {
        path: "/review",
        element: <Review/>
    }
])

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
    <React.StrictMode>
        <RouterProvider router={router}/>
    </React.StrictMode>
);
