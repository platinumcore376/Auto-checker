import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function LoginPage() {
    const navigate = useNavigate();

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleLogin = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            const response = await fetch('http://localhost:8000/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include', // Receives HttpOnly cookie from server
                body: JSON.stringify({ username, password }),
            });

            if (response.ok) {
                navigate('/home');
            } else {
                const data = await response.json().catch(() => ({}));
                setError(data.detail || 'Invalid username or password.');
            }
        } catch (err) {
            setError('Cannot reach authentication server.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-screen h-screen flex items-center justify-center bg-white">
            <div className="flex bg-white h-3/4 w-1/2 rounded-2xl shadow-xl border border-gray-100">
                <div className="h-full w-1/2 bg-dy-red rounded-l-2xl">
                    <form onSubmit={handleLogin} className="flex flex-col justify-center items-center h-full text-black px-6">
                        <h1 className="text-4xl font-bold mb-6 text-white">Login</h1>
                        <input
                            type="email"
                            placeholder="Email address"
                            required
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            className="w-full h-11 mb-3 border border-gray-300 rounded-md px-3 focus:outline-none focus:ring-2 focus:ring-dy-peach"
                        />
                        <input
                            type="password"
                            placeholder="Password"
                            required
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            className="w-full h-11 mb-4 border border-gray-300 rounded-md px-3 focus:outline-none focus:ring-2 focus:ring-dy-peach"
                        />
                        <button
                            type="submit"
                            disabled={loading}
                            className={`w-full h-11 bg-dy-peach font-semibold text-lg text-dy-red rounded-md transition-colors duration-200 ${
                                loading ? "opacity-50 cursor-not-allowed" : "hover:text-white hover:bg-opacity-90"
                            }`}
                        >
                            {loading ? "Authenticating..." : "Login"}
                        </button>
                        {error && (
                            <p className="pt-3 text-white text-sm text-center font-medium bg-red-800 bg-opacity-40 px-3 py-1 rounded mt-2 w-full">
                                {error}
                            </p>
                        )}
                    </form>
                </div>
                <div className="h-full w-1/2 rounded-r-2xl bg-login flex items-center justify-center">
                    <h1 className="text-dy-red font-bold text-3xl">AutoChecker</h1>
                </div>
            </div>
        </div>
    );
}